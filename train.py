import os
import torch
import logging

import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import ChangeDetectionConfig
from dataset import ChangeDetectionDataset
from evaluation import ChangeDetectionEvaluation

from visualizations import combine_masks, plot_confusion_matrix, plot_to_image


class ChangeDetectionTrain:
    def __init__(self):
        # read config
        self.config = ChangeDetectionConfig().get_config()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig()
        logging.root.setLevel(logging.INFO)

        self.logger.info('Logger was initialized')

        # init parameters
        self.train_dataloader, self.valid_dataloader = None, None

        self.model = None

        self.train_step, self.valid_step = 0, 0

        self.best_f1, self.best_recall, self.best_precision, self.best_accuracy = 0, 0, 0, 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create model
        # init dataloaders
        self.init_dataloaders()

        self.init_model()

        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        if self.config['resume_training']:
            self.resume_training()

        self.cd_evaluation = ChangeDetectionEvaluation(self.device, self.best_f1, self.best_recall,
                                                       self.best_precision, self.best_accuracy)

        os.makedirs(self.config['output_path'], exist_ok=True)
        os.makedirs(self.config['tensorboard_path'], exist_ok=True)

        # initialize tensorboard
        self.writer = SummaryWriter(self.config['tensorboard_path'])

    def init_dataloaders(self):
        train_dataset = ChangeDetectionDataset(dataset_path=self.config['dataset_path'],
                                               masks_path=self.config['masks_path'],
                                               df_path=self.config['train_df_path'],
                                               no_of_crops_per_combination=self.config['no_of_crops_per_combination'],
                                               training_mode=True,
                                               crop_size=self.config['crop_size'],
                                               return_masks=self.config['return_masks'])

        valid_dataset = ChangeDetectionDataset(dataset_path=self.config['dataset_path'],
                                               masks_path=self.config['masks_path'],
                                               df_path=self.config['valid_df_path'],
                                               training_mode=False,
                                               return_masks=self.config['return_masks'])

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.config['batch_size'],
                                           shuffle=True,
                                           num_workers=self.config['no_of_workers'])

        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=self.config['valid_batch'],
                                           shuffle=False,
                                           num_workers=self.config['valid_workers'])

        self.logger.info('Finished initialization of dataloaders')

    def init_model(self):
        self.model = smp.create_model(arch='unet', activation='sigmoid',
                                      in_channels=6,
                                      encoder_depth=4,
                                      decoder_channels=(256, 128, 64, 32))

        self.model.to(self.device)

    def resume_training(self):
        checkpoint = torch.load(self.config['model_path'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_step, self.valid_step = checkpoint['train_step'], checkpoint['valid_step']

        self.best_f1, self.best_recall, self.best_precision, self.best_accuracy = (checkpoint['best_f1'],
                                                                                   checkpoint['best_recall'],
                                                                                   checkpoint['best_precision'],
                                                                                   checkpoint['best_accuracy'])

        self.logger.info('Model resumed successfully: {}'.format(self.config['model_path']))

    def run(self):
        for epoch in range(self.config['no_of_epochs']):
            self.logger.info('Epoch {}/{}'.format(epoch + 1, self.config['no_of_epochs']))

            train_loss = self.train()
            valid_loss = self.validate(epoch)

            self.cd_evaluation.update_best()
            self.log_to_tensorboard(train_loss, valid_loss, epoch)

            self.logger.info('train loss: {}, valid loss: {}'.format(train_loss,
                                                                     valid_loss))

            if self.config['save_model']:
                self.save_model(epoch, train_loss, valid_loss)
        self.writer.flush()
        self.writer.close()

    def log_to_tensorboard(self, train_loss, valid_loss, epoch):
        self.writer.add_scalar("Loss_epoch/train", train_loss, epoch + 1)
        self.writer.add_scalar("Loss_epoch/valid", valid_loss, epoch + 1)
        self.writer.add_scalar("valid/accuracy", self.cd_evaluation.get_accuracy().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/precision", self.cd_evaluation.get_precision().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/recall", self.cd_evaluation.get_recall().cpu().numpy(), epoch + 1)
        self.writer.add_scalar("valid/f1", self.cd_evaluation.get_f1().cpu().numpy(), epoch + 1)

    def train(self):
        self.model.train()
        train_loss = 0

        for images_1, images_2, changes in tqdm(self.train_dataloader, desc='Training'):
            images = torch.concat([images_1, images_2], dim=1)
            images = images.to(self.device)
            changes = changes.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, changes)

            train_loss += loss.item()

            self.writer.add_scalar("Loss_step/train", loss.item(), self.train_step)
            self.train_step += 1
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

        return train_loss / len(self.train_dataloader)

    def validate(self, epoch):
        self.model.eval()
        self.cd_evaluation.reset()
        valid_loss = 0

        with torch.no_grad():
            for images_1, images_2, changes in tqdm(self.valid_dataloader, desc='Validation'):
                images = torch.concat([images_1, images_2], dim=1)
                images = images.to(self.device)
                changes = changes.to(self.device)

                outputs = self.model(images)

                loss = self.loss_fn(outputs, changes)
                self.cd_evaluation.update(ground_truth=changes, prediction=outputs)
                self.writer.add_scalar("Loss_step/valid", loss.item(), self.valid_step)

                self.writer.add_image('confusion_matrix',
                                      plot_to_image(plot_confusion_matrix(self.cd_evaluation.get_confusion_matrix(),
                                                                          ['change', 'background'])),
                                      epoch + 1,
                                      dataformats='HWC')
                self.writer.add_image('example_1', combine_masks(changes[0], outputs[0] > 0.5), epoch + 1,
                                      dataformats='HWC')

                self.valid_step += 1
                valid_loss += loss.item()

        return valid_loss / len(self.valid_dataloader)

    def save_model(self, epoch, train_loss, valid_loss):
        # save the model
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_losses': train_loss,
            'validation_losses': valid_loss,
            'train_step': self.train_step,
            'valid_step': self.valid_step,
            'best_f1': self.cd_evaluation.best_f1,
            'best_recall': self.cd_evaluation.best_recall,
            'best_precision': self.cd_evaluation.best_precision,
            'best_accuracy': self.cd_evaluation.best_accuracy
        }

        torch.save(checkpoint, os.path.join(self.config['output_path'], 'model_{:03d}'.format(epoch + 1)))

        if self.cd_evaluation.is_best_f1:
            torch.save(checkpoint, os.path.join(self.config['output_path'], 'best_f1_{:03d}'.format(epoch + 1)))

        if self.cd_evaluation.is_best_recall:
            torch.save(checkpoint, os.path.join(self.config['output_path'], 'best_recall_{:03d}'.format(epoch + 1)))

        if self.cd_evaluation.is_best_precision:
            torch.save(checkpoint,
                       os.path.join(self.config['output_path'], 'best_precision_{:03d}'.format(epoch + 1)))

        if self.cd_evaluation.is_best_accuracy:
            torch.save(checkpoint,
                       os.path.join(self.config['output_path'], 'best_accuracy_{:03d}'.format(epoch + 1)))


if __name__ == '__main__':
    ChangeDetectionTrain().run()
