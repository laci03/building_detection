import cv2
import torch

import numpy as np
import pandas as pd
import rasterio as rio
import albumentations as A

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import Resize, InterpolationMode


class BuildingDetectionDataset(Dataset):
    def __init__(self, dataset_path, masks_path, df_path, no_of_crops_per_combination=1,
                 training_mode=True, crop_size=None, shuffle=False, transform=None,
                 rgb_mean=None, rgb_std=None):
        self.dataset_path = Path(dataset_path)
        self.masks_path = Path(masks_path)

        self.shuffle = shuffle

        self.df = pd.read_csv(df_path, index_col=0)

        self.no_of_crops_per_combination = no_of_crops_per_combination

        self.training_mode = training_mode

        self.file_list = self.create_filelist()
        if self.shuffle:
            np.random.shuffle(self.file_list)

        self.crop_size = crop_size

        self.transform = transform

        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self.img_size = self.crop_size if self.training_mode else 1024

        self.resize_img = Resize(size=self.img_size*2)
        self.resize_mask = Resize(size=self.img_size*2, interpolation=InterpolationMode.NEAREST)

    def create_filelist(self):
        ds = []

        for aoi_name in sorted(list(set(self.df['AOI_name']))):
            file_list = sorted(list(set(self.df[self.df['AOI_name'] == aoi_name]['filename'])))
            for file in file_list:
                    ds.append((aoi_name, file))

        return ds

    def get_image_and_mask(self, AOI_name, file_name):
        image_path = self.dataset_path / AOI_name / 'images_masked' / '{}.tif'.format(file_name)
        mask_path = self.masks_path / AOI_name / '{}.png'.format(file_name)

        r = rio.open(image_path).read()

        image = r.transpose((1, 2, 0))[:, :, :-1]

        mask = np.array(np.bool_(cv2.imread(str(mask_path), 0))[:, :, None], np.float32)
        return image, mask

    @staticmethod
    def update_mask(mask, img_shape):
        new_mask = np.zeros((img_shape[0], img_shape[1], 1), dtype=bool)
        new_mask[:mask.shape[0], :mask.shape[1], :] = mask
        return new_mask

    def __getitem__(self, idx):
        if self.training_mode:
            idx = idx // self.no_of_crops_per_combination

        aoi_name, file_name = self.file_list[idx]

        image, mask = self.get_image_and_mask(aoi_name, file_name)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)

            image, mask = transformed['image'], transformed['mask']

        if self.training_mode:  # crop from the initial image
            x = np.random.randint(0, image.shape[0] - self.crop_size)
            y = np.random.randint(0, image.shape[1] - self.crop_size)

            image = image[x:x + self.crop_size, y:y + self.crop_size, :]
            mask = mask[x:x + self.crop_size, y:y + self.crop_size, :]
        else:
            height, width = image.shape[:-1]

            pad_height = 0
            pad_width = 0
            if width % 8:
                pad_width = 8 - width % 8
            if height % 8:
                pad_height = 8 - height % 8

            if pad_width or pad_height:
                image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                mask = self.update_mask(mask, image.shape[:-1])

        if self.rgb_mean is not None and self.rgb_std is not None:
            image = (image - self.rgb_mean) / self.rgb_std

        image = torch.from_numpy(np.array(image.transpose((2, 0, 1)), dtype=np.float32))
        mask = torch.from_numpy(np.array(mask.transpose((2, 0, 1)), dtype=np.float32))

        return self.resize_img(image), self.resize_mask(mask)

    def __len__(self):
        if self.training_mode:
            return self.no_of_crops_per_combination * len(self.file_list)
        else:
            return len(self.file_list)

    def train(self):
        self.training_mode = True

    def valid_test(self):
        self.training_mode = False


def visualize_dataset(image, mask, resize_factor=2, rgb_mean=None, rgb_std=None):
    image = image.numpy().transpose((1, 2, 0))[:, :, ::-1]

    if rgb_mean is not None and rgb_std is not None:
        image = image * rgb_std[::-1] + rgb_mean[::-1]

    image = np.array(image, dtype=np.uint8)
    mask = mask.numpy().transpose((1, 2, 0))

    dims = (image.shape[1] // resize_factor, image.shape[0] // resize_factor)

    cv2.imshow('image', cv2.resize(image, dims))
    cv2.imshow('mask', cv2.resize(mask, dims))

    return cv2.waitKey()


if __name__ == '__main__':
    rgb_mean = (120.63812214, 105.92798168, 77.53151193)
    rgb_std = (60.0614334, 47.96735684, 44.21755486)

    resize_factor = 1
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(scale_limit=(-0.1, 0.1))
    ])

    dataset = BuildingDetectionDataset(dataset_path='../change_detection_dataset/SN7_buildings/train',
                                       masks_path='../change_detection_dataset/SN7_masks',
                                       df_path='../change_detection_dataset/dataset_1/train.csv',
                                       no_of_crops_per_combination=10,
                                       training_mode=False,
                                       crop_size=256,
                                       shuffle=False,
                                       transform=transform,
                                       rgb_mean=rgb_mean,
                                       rgb_std=rgb_std)

    for idx in tqdm(range(len(dataset))):
        ch = visualize_dataset(*dataset[idx], resize_factor=resize_factor, rgb_mean=rgb_mean, rgb_std=rgb_std)

        if ch & 0xff == ord('q'):
            break
