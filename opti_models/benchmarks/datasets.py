import typing as t
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2


class ImagenetDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, in_size: t.Tuple = (224, 224), mode: str = 'torch'):
        self.mode = mode
        self.data_df = data_df
        if "names" in data_df.columns:
            self.file_names = data_df['names']
        else:
            raise ValueError(
                "'names' not a name of labels._df columns"
            )

        self.albu_augmentations = Compose([
            Resize(
                height=in_size[0],
                width=in_size[1],
                interpolation=cv2.INTER_AREA
            ),
            Normalize(),
            ToTensorV2(),
        ], p=1)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        label = self.data_df[self.data_df['names'] == path]['labels'].values.tolist()[0]
        image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = self.albu_augmentations(image=image)['image']
        return image, label, path

    def collate_fn(self, batch: t.Tuple) -> t.Tuple:
        imgs = [x[0] for x in batch]
        ch, h, w = imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2]
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, ch, h, w)
        for i in range(num_imgs):
            inputs[i] = imgs[i]
        labels = [x[1] for x in batch]
        paths = [x[2] for x in batch]
        return inputs, labels, paths