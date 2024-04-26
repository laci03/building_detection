import os
import torch
import logging
import time
import gc
import shutil

import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.losses import DiceLoss
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import BuildingDetectionConfig
from dataset import BuildingDetectionDataset
from evaluation import BuildingDetectionEvaluation

from visualizations import combine_masks, plot_confusion_matrix, plot_to_image


class BuildingDetectionTrain:
    def __init__(self, config_path):
        # read config
        self.config = BuildingDetectionConfig(config_path).get_config()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)

        self.logger.info('The logger initialized successfully')

        # init parameters
        self.train_dataloader, self.valid_dataloader = None, None

        self.model = None

        self.train_step, self.valid_step, self.start_epoch = 0, 0, 0

        self.best_f1, self.best_recall, self.best_precision, self.best_accuracy = 0, 0, 0, 0

        self.best_valid_loss = np.inf
        self.early_stop_counter = 0

        self.batches_to_plot = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create model
        # init dataloaders
        self.init_dataloaders()

        self.init_model()

        # self.loss_fn = torch.nn.BCELoss()
        self.loss_fn = DiceLoss(mode='binary', from_logits=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        if self.config['resume_training']:
            self.resume_training()

        self.bd_evaluation = BuildingDetectionEvaluation(self.device, self.best_f1, self.best_recall,
                                                         self.best_precision, self.best_accuracy)

        os.makedirs(self.config['output_path'], exist_ok=True)
        os.makedirs(self.config['tensorboard_path'], exist_ok=True)

        # initialize tensorboard
        self.writer = SummaryWriter(self.config['tensorboard_path'])

        self.print_status()

        shutil.copy(config_path, self.config['output_path'])

    def print_status(self):
        self.logger.info('Train batch size: {}'.format(self.config['batch_size']))
        self.logger.info('Validation batch size: {}'.format(self.config['valid_batch']))
        self.logger.info('Image resize: {}'.format(self.config['image_resize']))

        self.logger.info('No of steps in training: {}'.format(len(self.train_dataloader)))
        self.logger.info('No of steps in validation: {}'.format(len(self.valid_dataloader)))
        self.logger.info('Cuda available: {}'.format(torch.cuda.is_available()))

    def init_dataloaders(self):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=(-0.1, 0.1))
        ])

        train_dataset = BuildingDetectionDataset(dataset_path=self.config['dataset_path'],
                                                 masks_path=self.config['masks_path'],
                                                 df_path=self.config['train_df_path'],
                                                 no_of_crops_per_combination=self.config['no_of_crops_per_combination'],
                                                 training_mode=True,
                                                 crop_size=self.config['crop_size'],
                                                 transform=transform,
                                                 rgb_mean=self.config['rgb_mean'],
                                                 rgb_std=self.config['rgb_std'],
                                                 image_resize=self.config['image_resize'])

        valid_dataset = BuildingDetectionDataset(dataset_path=self.config['dataset_path'],
                                                 masks_path=self.config['masks_path'],
                                                 df_path=self.config['valid_df_path'],
                                                 training_mode=True,
                                                 crop_size=256,
                                                 no_of_crops_per_combination=16,
                                                 shuffle=True,
                                                 rgb_mean=self.config['rgb_mean'],
                                                 rgb_std=self.config['rgb_std'],
                                                 image_resize=self.config['image_resize'])

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.config['batch_size'],
                                           shuffle=True,
                                           num_workers=self.config['no_of_workers'])

        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=self.config['valid_batch'],
                                           shuffle=False,
                                           num_workers=self.config['valid_workers'])

        self.batches_to_plot = np.random.choice(range(len(self.valid_dataloader)),
                                                self.config['no_of_batches_to_plot'],
                                                replace=False)

        self.logger.info('The dataloaders initialized successfully')

    def init_model(self):
        self.model = smp.create_model(arch='unet', activation='sigmoid',
                                      encoder_name=self.config['backbone'],
                                      in_channels=3)

        self.model.to(self.device)

    def resume_training(self):
        checkpoint = torch.load(self.config['model_path'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.train_step, self.valid_step, self.start_epoch = (checkpoint['train_step'],
                                                              checkpoint['valid_step'],
                                                              checkpoint['epoch'])

        self.best_f1, self.best_recall, self.best_precision, self.best_accuracy = (checkpoint['best_f1'],
                                                                                   checkpoint['best_recall'],
                                                                                   checkpoint['best_precision'],
                                                                                   checkpoint['best_accuracy'])

        self.logger.info('Model resumed successfully: {}'.format(self.config['model_path']))

    def early_stop(self, valid_loss):
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.early_stop_counter = 0
        elif valid_loss > (self.best_valid_loss + self.config['min_delta']):
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.config['patience']

    def run(self):
        for epoch in range(self.start_epoch, self.config['no_of_epochs'], 1):
            start_time = time.time()

            train_loss = self.train()
            valid_loss = self.validate(epoch)
            self.scheduler.step(valid_loss)

            self.bd_evaluation.update_best()
            early_stop = self.early_stop(valid_loss)

            self.log_to_tensorboard(train_loss, valid_loss, epoch)

            if self.config['save_model']:
                self.save_model(epoch)

            self.logger.info('[{}/{}] train loss: {}, valid loss: {}, elapsed time: {}'.format(epoch + 1,
                                                                                               self.config[
                                                                                                   'no_of_epochs'],
                                                                                               train_loss,
                                                                                               valid_loss,
                                                                                               time.time() - start_time))

            if early_stop:
                self.logger.info('Early stop activated, min_delta: {}, patience: {}'.format(self.config['min_delta'],
                                                                                            self.config['patience']))
                break

            gc.collect()

        self.writer.flush()
        self.writer.close()

    def log_to_tensorboard(self, train_loss, valid_loss, epoch):
        self.writer.add_scalar("Loss_epoch/train", train_loss, epoch + 1)
        self.writer.add_scalar("Loss_epoch/valid", valid_loss, epoch + 1)
        self.writer.add_scalar("params/patience", self.early_stop_counter, epoch + 1)
        self.writer.add_scalar("params/learning_rate", self.scheduler.get_last_lr()[0], epoch + 1)
        self.writer.add_scalar("valid/accuracy", self.bd_evaluation.get_accuracy().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/precision", self.bd_evaluation.get_precision().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/recall", self.bd_evaluation.get_recall().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/f1", self.bd_evaluation.get_f1().cpu().numpy(), epoch + 1)

    def train(self):
        self.model.train()
        train_loss = 0

        for images, masks in self.train_dataloader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, masks)

            train_loss += loss.item()

            self.writer.add_scalar("Loss_step/train", loss.item(), self.train_step)
            self.train_step += 1
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

        return train_loss / len(self.train_dataloader)

    def validate(self, epoch):
        self.model.eval()
        self.bd_evaluation.reset()
        valid_loss = 0

        with torch.no_grad():
            for i, (images, masks) in enumerate(self.valid_dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)

                loss = self.loss_fn(outputs, masks)
                self.bd_evaluation.update(ground_truth=masks, prediction=outputs)
                self.writer.add_scalar("Loss_step/valid", loss.item(), self.valid_step)

                if i in self.batches_to_plot:
                    initial_index = np.where(i == self.batches_to_plot)[0][0] * self.config['valid_batch']
                    for index in range(self.config['valid_batch']):
                        self.writer.add_image('example_{}'.format(initial_index + index), combine_masks(masks[index],
                                                                                                        outputs[
                                                                                                            index] > 0.5),
                                              epoch + 1,
                                              dataformats='HWC')

                self.valid_step += 1
                valid_loss += loss.item()

            self.writer.add_image('confusion_matrix',
                                  plot_to_image(plot_confusion_matrix(self.bd_evaluation.get_confusion_matrix(),
                                                                      ['building', 'background'])),
                                  epoch + 1,
                                  dataformats='HWC')

        return valid_loss / len(self.valid_dataloader)

    def save_model(self, epoch):
        # save the model
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'train_step': self.train_step,
            'valid_step': self.valid_step,
            'scheduler': self.scheduler.state_dict(),
            'best_f1': self.bd_evaluation.best_f1,
            'best_recall': self.bd_evaluation.best_recall,
            'best_precision': self.bd_evaluation.best_precision,
            'best_accuracy': self.bd_evaluation.best_accuracy
        }

        torch.save(checkpoint, os.path.join(self.config['output_path'], 'model'))

        if self.bd_evaluation.is_best_f1:
            torch.save(checkpoint, os.path.join(self.config['output_path'], 'best_f1'))

        if self.bd_evaluation.is_best_recall:
            torch.save(checkpoint, os.path.join(self.config['output_path'], 'best_recall'))

        if self.bd_evaluation.is_best_precision:
            torch.save(checkpoint,
                       os.path.join(self.config['output_path'], 'best_precision'))

        if self.bd_evaluation.is_best_accuracy:
            torch.save(checkpoint,
                       os.path.join(self.config['output_path'], 'best_accuracy'))


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver('../building_detection_experiments/sacred'))


@ex.config
def cd_config():
    config_path = 'config.ini'


@ex.main
def cd_main(config_path):
    BuildingDetectionTrain(config_path).run()


if __name__ == '__main__':
    ex.run_commandline()
