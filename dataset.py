import cv2
import pandas as pd
import rasterio as rio

from pathlib import Path

from tqdm import tqdm
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):
    def __init__(self, dataset_path, masks_path, df_path):
        self.dataset_path = Path(dataset_path)
        self.masks_path = Path(masks_path)

        self.df = pd.read_csv(df_path, index_col=0)

        self.image_and_mask_dict = self.get_image_and_mask()

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
        pass


if __name__ == '__main__':
    dataset_path = '../change_detection_dataset/SN7_buildings/train'
    masks_path = '../change_detection_dataset/SN7_masks'
    df_path = '../change_detection_dataset/dataset_1/train.csv'

    dataset = ChangeDetectionDataset(dataset_path, masks_path, df_path)

    dataset[0]