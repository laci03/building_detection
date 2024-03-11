import cv2
import torch

import numpy as np
import pandas as pd
import rasterio as rio

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):
    def __init__(self, dataset_path, masks_path, df_path,
                 no_of_crops_per_combination, training_mode=True, crop_size=None):
        self.dataset_path = Path(dataset_path)
        self.masks_path = Path(masks_path)

        self.df = pd.read_csv(df_path, index_col=0)

        # read all dataset into memory
        self.image_and_mask_dict = self.get_image_and_mask()

        self.no_of_crops_per_combination = no_of_crops_per_combination

        self.training_mode = training_mode

        self.combinations_list = self.create_combinations()
        self.crop_size = crop_size

    def create_combinations(self):
        combinations = []

        for aoi_name, file_list in sorted(self.image_and_mask_dict.items()):
            for i in range(len(file_list[:-1])):
                for j in range(i+1, len(file_list)):
                    combinations.append((aoi_name, i, j))

        return combinations

    def get_image_and_mask(self):
        AOI_names = list(set(self.df['AOI_name']))
        image_and_mask_dict = {}

        for AOI_name in tqdm(AOI_names):
            file_names = sorted(list(set(self.df[self.df['AOI_name'] == AOI_name]['filename'])))
            images = []

            for file_name in file_names:
                image_path = self.dataset_path / AOI_name / 'images_masked' / '{}.tif'.format(file_name)
                mask_path = self.masks_path / AOI_name / '{}.png'.format(file_name)

                r = rio.open(image_path).read()

                image = r.transpose((1, 2, 0))[:, :, :-1]

                mask = cv2.imread(str(mask_path), 0)

                images.append((file_name, image, mask))

            image_and_mask_dict[AOI_name] = images

        return image_and_mask_dict

    def __getitem__(self, idx):
        if self.training_mode:
            idx = idx // self.no_of_crops_per_combination

        combination = self.combinations_list[idx]

        file_name_1, image_1, mask_1 = self.image_and_mask_dict[combination[0]][combination[1]]
        file_name_2, image_2, mask_2 = self.image_and_mask_dict[combination[0]][combination[2]]

        udm_mask = np.logical_not(np.logical_or(np.all(image_1 == 0, axis=-1),
                                                np.all(image_2 == 0, axis=-1)))[:, :, None]

        image_1 = udm_mask * image_1
        image_2 = udm_mask * image_2

        mask_1 = udm_mask * np.bool_(mask_1[:, :, None])
        mask_2 = udm_mask * np.bool_(mask_2[:, :, None])

        change = np.logical_xor(mask_1, mask_2)

        if self.training_mode:  # crop from the initial image
            x = np.random.randint(0, image_1.shape[0] - self.crop_size)
            y = np.random.randint(0, image_1.shape[1] - self.crop_size)

            image_1 = image_1[x:x + self.crop_size, y:y + self.crop_size, :]
            image_2 = image_2[x:x + self.crop_size, y:y + self.crop_size, :]
            mask_1 = mask_1[x:x + self.crop_size, y:y + self.crop_size, :]
            mask_2 = mask_2[x:x + self.crop_size, y:y + self.crop_size, :]
            change = change[x:x + self.crop_size, y:y + self.crop_size, :]

        image_1 = torch.from_numpy(image_1.transpose((2, 0, 1)) / 255.0)
        image_2 = torch.from_numpy(image_2.transpose((2, 0, 1)) / 255.0)

        mask_1 = torch.from_numpy((mask_1 * 1.0).transpose((2, 0, 1)))
        mask_2 = torch.from_numpy((mask_2 * 1.0).transpose((2, 0, 1)))

        change = torch.from_numpy(change.transpose((2, 0, 1)) * 1.0)
        return image_1, image_2, mask_1, mask_2, change

    def __len__(self):
        if self.training_mode:
            return self.no_of_crops_per_combination * len(self.combinations_list)
        else:
            return len(self.combinations_list)

    def train(self):
        self.training_mode = True

    def valid_test(self):
        self.training_mode = False


def visualize_dataset(image_1, image_2, mask_1, mask_2, change, resize_factor=2):
    image_1 = image_1.numpy().transpose((1, 2, 0))[:, :, ::-1]
    image_2 = image_2.numpy().transpose((1, 2, 0))[:, :, ::-1]

    image = np.hstack([image_1, image_2])

    mask_1 = mask_1.numpy().transpose((1, 2, 0))
    mask_2 = mask_2.numpy().transpose((1, 2, 0))

    mask = np.hstack([mask_1, mask_2])

    change = change.numpy().transpose((1, 2, 0))

    dims = (image.shape[1]//resize_factor, image.shape[0]//resize_factor)
    change_dims = (change.shape[1]//resize_factor, change.shape[0]//resize_factor)

    cv2.imshow('image', cv2.resize(image, dims))
    cv2.imshow('mask', cv2.resize(mask, dims))
    cv2.imshow('change', cv2.resize(change, change_dims))

    return cv2.waitKey()


if __name__ == '__main__':
    resize_factor = 1

    dataset = ChangeDetectionDataset(dataset_path='../change_detection_dataset/SN7_buildings/train',
                                     masks_path='../change_detection_dataset/SN7_masks',
                                     df_path='../change_detection_dataset/dataset_1/valid.csv',
                                     no_of_crops_per_combination=10,
                                     training_mode=True,
                                     crop_size=256)

    for idx in tqdm(range(len(dataset))):
        ch = visualize_dataset(*dataset[idx], resize_factor=resize_factor)

        if ch & 0xff == ord('q'):
            break
