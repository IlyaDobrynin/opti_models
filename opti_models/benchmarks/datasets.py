import typing as t
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImagenetDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, in_size: int = 224):
        self.data_df = data_df
        if "names" in data_df.columns:
            self.file_names = data_df['names']
        else:
            raise ValueError(
                "'names' not a name of labels._df columns"
            )

        self.augmentations = transforms.Compose([
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = self.file_names[idx]
        label = self.data_df[self.data_df['names'] == path]['labels'].values.tolist()[0]
        image = Image.open(path)
        image = self.augmentations(img=image)
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