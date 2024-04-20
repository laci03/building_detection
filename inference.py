import cv2

import torch
import argparse

import numpy as np
import segmentation_models_pytorch as smp

from pathlib import Path

from dataset import BuildingDetectionDataset
from visualizations import combine_masks
from config import BuildingDetectionConfig


def visualize_results(image_1, image_2, mask_1, mask_2, change, pred, resize_factor=2, rgb_mean=(0, 0, 0),
                      rgb_std=(1, 1, 1)):
    image_1 = image_1.numpy().transpose((1, 2, 0))
    image_2 = image_2.numpy().transpose((1, 2, 0))

    image = np.hstack([image_1, image_2]) * rgb_std + rgb_mean
    image = image[:, :, ::-1]

    new_mask = combine_masks(mask_1, mask_2)
    new_change = combine_masks(change, pred > 0.5)

    dims = (image.shape[1] // resize_factor, image.shape[0] // resize_factor)
    change_dims = (new_change.shape[1] // resize_factor, new_change.shape[0] // resize_factor)

    cv2.imshow('image', cv2.resize(np.array(image, dtype=np.uint8), dims))

    cv2.imshow('mask', cv2.resize(new_mask, change_dims))
    cv2.imshow('cd', cv2.resize(new_change, change_dims))

    return cv2.waitKey()


def run_inference(config):
    save_output = bool(config['output_path'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    if config['input_path'].suffix == '.csv':
        inference_dataset = BuildingDetectionDataset(dataset_path=config['dataset_path'],
                                                     masks_path=config['masks_path'],
                                                     df_path=str(config['input_path']),
                                                     return_masks=True,
                                                     training_mode=False,
                                                     shuffle=False,
                                                     rgb_mean=config['rgb_mean'],
                                                     rgb_std=config['rgb_std'])

    # load model
    model = smp.create_model(arch='unet', activation='sigmoid',
                             in_channels=6,
                             encoder_depth=4,
                             decoder_channels=(256, 128, 64, 32))

    checkpoint = torch.load(config['model_path'])

    model.load_state_dict(checkpoint['model'])
    model.to(device)

    with torch.no_grad():
        model.eval()
        for image_1, image_2, mask_1, mask_2, change in iter(inference_dataset):
            image = torch.concat([image_1, image_2], dim=0)
            image = image.to(device)[None, :, :, :]

            output = model(image)[0].cpu()

            ch = visualize_results(image_1, image_2, mask_1, mask_2, change, output, config['resize_factor'],
                                   config['rgb_mean'], config['rgb_std'])

            if ch & 0xff == ord('q'):
                return 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='', type=Path)
    parser.add_argument('--output_path', default='', type=Path)

    parser.add_argument('--dataset_path', default='', type=Path)
    parser.add_argument('--masks_path', default='', type=Path)

    parser.add_argument('--model_path', default='../change_detection_output/exp_13/best_f1_052', type=Path)  # exp_2 66
    parser.add_argument('--resize_factor', default=2, type=int)

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    config_path = Path(args['model_path']).parent / 'config.ini'
    config = BuildingDetectionConfig(config_path).get_config()

    config['model_path'] = args['model_path']
    config['resize_factor'] = args['resize_factor']

    if args['input_path'] != Path(''):
        config['input_path'] = args['input_path']
    else:
        config['input_path'] = Path(config['valid_df_path'])

    if args['output_path'] != Path(''):
        config['output_path'] = args['output_path']

    if args['dataset_path'] != Path(''):
        config['dataset_path'] = args['dataset_path']

    if args['masks_path'] != Path(''):
        config['mask_path'] = args['mask_path']

    run_inference(config)
