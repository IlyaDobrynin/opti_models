import typing as t

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.image_utils import imagenet_preprocess


class ImagenetDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, in_size: t.Tuple = (224, 224), mode: str = 'torch'):
        self.mode = mode
        self.data_df = data_df
        if "names" in data_df.columns:
            self.file_names = data_df['names']
        else:
            raise ValueError("'names' not a name of labels._df columns")
        self.in_size = in_size

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        label = self.data_df[self.data_df['names'] == path]['labels'].values.tolist()[0]
        image = imagenet_preprocess(image_path=path, size=(self.in_size[0], self.in_size[1]))
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
