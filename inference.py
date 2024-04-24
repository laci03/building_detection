import cv2
import os

import torch
import argparse

import numpy as np
import segmentation_models_pytorch as smp

from pathlib import Path

from dataset import BuildingDetectionDataset
from visualizations import combine_masks
from config import BuildingDetectionConfig
from tqdm import tqdm


def visualize_results(image, mask, pred, resize_factor=2, rgb_mean=(0, 0, 0),
                      rgb_std=(1, 1, 1), output_img_path=None):
    image = image.cpu().numpy().transpose((1, 2, 0)) * rgb_std + rgb_mean
    image = image[:, :, ::-1]

    new_mask = combine_masks(mask, pred > 0.5)

    dims = (image.shape[1] // resize_factor, image.shape[0] // resize_factor)
    change_dims = (new_mask.shape[1] // resize_factor, new_mask.shape[0] // resize_factor)

    if config['save_image']:
        cv2.imwrite(output_img_path, new_mask)

    if config['show_image']:
        cv2.imshow('image', cv2.resize(np.array(image, dtype=np.uint8), dims))

        cv2.imshow('mask', cv2.resize(new_mask, change_dims))

        return cv2.waitKey(0)
    else:
        return 0


def run_inference(config):
    if config['save_image']:
        os.makedirs(config['output_path'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    if config['input_path'].suffix == '.csv':
        inference_dataset = BuildingDetectionDataset(dataset_path=config['dataset_path'],
                                                     masks_path=config['masks_path'],
                                                     df_path=str(config['input_path']),
                                                     training_mode=False,
                                                     shuffle=False,
                                                     rgb_mean=config['rgb_mean'],
                                                     rgb_std=config['rgb_std'],
                                                     image_resize=config['image_resize'])

    # load model
    model = smp.create_model(arch='unet', activation='sigmoid',
                             encoder_name='resnet50',
                             in_channels=3)

    checkpoint = torch.load(config['model_path'])

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    with torch.no_grad():
        model.eval()
        for i, (image, mask) in tqdm(enumerate(iter(inference_dataset))):
            image = image.to(device)[None, :, :, :]

            output = model(image)[0].cpu()

            output_image_path = os.path.join(config['output_path'],
                                             '{}.png'.format('_'.join(inference_dataset.file_list[i])))

            ch = visualize_results(image[0], mask, output, config['resize_factor'],
                                   config['rgb_mean'], config['rgb_std'], output_image_path)

            if ch & 0xff == ord('q'):
                return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='', type=Path)
    parser.add_argument('--output_path', default='', type=Path)

    parser.add_argument('--dataset_path', default='', type=Path)
    parser.add_argument('--masks_path', default='', type=Path)

    parser.add_argument('--model_path', default='../building_detection_experiments/exp_004/best_f1',
                        type=Path)
    parser.add_argument('--resize_factor', default=1, type=int)

    parser.add_argument('--show_image', default=False, type=bool)
    parser.add_argument('--save_image', default=True, type=bool)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    config_path = Path(args['model_path']).parent / 'config.ini'
    config = BuildingDetectionConfig(config_path).get_config()

    config['model_path'] = args['model_path']
    config['resize_factor'] = args['resize_factor']
    config['show_image'] = args['show_image']
    config['save_image'] = args['save_image']

    if args['input_path'] != Path(''):
        config['input_path'] = args['input_path']
    else:
        config['input_path'] = Path(config['valid_df_path'])

    if args['output_path'] != Path(''):
        config['output_path'] = args['output_path']
    else:
        config['output_path'] = os.path.join(config['output_path'], 'valid')

    if args['dataset_path'] != Path(''):
        config['dataset_path'] = args['dataset_path']

    if args['masks_path'] != Path(''):
        config['mask_path'] = args['mask_path']

    run_inference(config)
