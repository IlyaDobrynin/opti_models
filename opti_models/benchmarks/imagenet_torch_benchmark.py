import os
import ast
import typing as t
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import cv2
import logging
from torchsummary import summary
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from ido_cv import draw_images
from opti_models.models import models_facade
from opti_models.benchmarks.datasets import ImagenetDataset


class SimpleBenchmark:
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            workers: int,
            in_size: int = 224
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.workers = workers
        self.in_size = in_size

    def _prepare_data(
            self, path_to_images: str, path_to_labels: str, show: bool = False):
        with open(path_to_labels) as f:
            doc = f.read()
        doc = ast.literal_eval(doc)
        img_classes = [int(s) for s in os.listdir(path_to_images)]

        out_dict = {}
        for img_cls in tqdm(img_classes, total=len(img_classes)):
            images_folder_path = os.path.join(path_to_images, str(img_cls))
            for img_name in os.listdir(images_folder_path):
                path_to_image = os.path.join(images_folder_path, img_name)
                out_dict[path_to_image] = img_cls
                if show:
                    image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
                    title_dict = {'text': doc[img_cls], 'size': 14, 'weight': 'bold', 'color': 'black'}
                    draw_images([image], title=title_dict)

        out_df = pd.DataFrame()
        out_df['names'] = list(out_dict.keys())
        out_df['labels'] = list(out_dict.values())

        return out_df

    def _load_model(self, show: bool = False):
        models_facade_obj = models_facade.ModelFacade(task="backbones")
        model = models_facade_obj.get_model_class(model_definition=self.model_name)(requires_grad=False, pretrained='imagenet')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(device)
        if show:
            summary(model, input_size=(3, 224, 224))
        return model

    def _make_dataloader(self, data_df: pd.DataFrame):
        dataset_obj = ImagenetDataset(data_df=data_df, in_size=self.in_size)
        dataloader = DataLoader(
            dataset=dataset_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=dataset_obj.collate_fn,
            pin_memory=True
        )
        return dataloader

    @staticmethod
    def top_n_accuracy(preds: t.List, truths: t.List, n: int):
        best_n = np.argsort(-np.asarray(preds), axis=1)[:, :n]
        successes = 0
        for i, truth in enumerate(truths):
            if truth in best_n[i, :]:
                successes += 1
        return float(successes) / len(truths)

    def _compute_metrics(self, trues_df: pd.DataFrame, preds: t.Dict):
        true_labels = []
        pred_labels = []
        for name in preds.keys():
            true_labels.append(trues_df[trues_df['names'] == name]['labels'].values.tolist()[0])
            pred_labels.append(preds[name])

        top_1_acc = self.top_n_accuracy(preds=pred_labels, truths=true_labels, n=1)
        top_5_acc = self.top_n_accuracy(preds=pred_labels, truths=true_labels, n=5)
        logging.info(f"\tBENCHMARK DONE FOR {self.model_name}")
        logging.info(f"\tTOP 1 ACCURACY: {top_1_acc * 100:.2f}\tTOP 1 ERROR: {(1 - top_1_acc) * 100:.2f}")
        logging.info(f"\tTOP 5 ACCURACY: {top_5_acc * 100:.2f}\tTOP 5 ERROR: {(1 - top_5_acc) * 100:.2f}")

    def _inference_loop(self, dataloader: DataLoader, model: torch.nn.Module):
        preds_dict = {}
        avg_batch_time = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].cuda()
            names = batch[2]
            batch_time = time()
            preds = F.softmax(model(inputs), dim=-1).data.cpu().numpy()
            avg_batch_time.append(time() - batch_time)
            preds_dict.update({name: label for name, label in zip(names, preds)})
        logging.info(f"\tAverage fps: {self.batch_size / np.mean(avg_batch_time)}")
        return preds_dict

    def process(self, path_to_images: str, path_to_labels: str):
        labels_df = self._prepare_data(path_to_images=path_to_images, path_to_labels=path_to_labels)
        model = self._load_model()
        dataloader = self._make_dataloader(data_df=labels_df)
        preds_dict = self._inference_loop(dataloader=dataloader, model=model)
        self._compute_metrics(trues_df=labels_df, preds=preds_dict)


if __name__ == '__main__':
    path_to_images = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
    path_to_class_names = "/mnt/Disk_G/DL_Data/source/imagenet/imagenet1000_clsidx_to_labels.txt"
    model_name = 'mobilenetv3_large_w1'
    in_size = 224

    bench_obj = SimpleBenchmark(model_name=model_name, batch_size=64, workers=11, in_size=in_size)
    preds = bench_obj.process(path_to_images=path_to_images, path_to_labels=path_to_class_names)