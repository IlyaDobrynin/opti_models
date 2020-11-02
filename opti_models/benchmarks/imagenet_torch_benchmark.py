import os
import typing as t
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import logging
from torchsummary import summary
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.nn import functional as F
from opti_models.models import models_facade
from opti_models.benchmarks.datasets import ImagenetDataset
logging.basicConfig(level=logging.INFO)


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

    def _prepare_data(self, path_to_images: str):
        img_classes = [int(s) for s in os.listdir(path_to_images)]

        out_dict = {}
        for img_cls in img_classes:
            images_folder_path = os.path.join(path_to_images, str(img_cls))
            for img_name in os.listdir(images_folder_path):
                path_to_image = os.path.join(images_folder_path, img_name)
                out_dict[path_to_image] = img_cls

        out_df = pd.DataFrame()
        out_df['names'] = list(out_dict.keys())
        out_df['labels'] = list(out_dict.values())

        return out_df

    def _load_model(self, show: bool = False):
        models_facade_obj = models_facade.ModelFacade(task="backbones")
        model = models_facade_obj.get_model_class(
            model_definition=self.model_name
        )(requires_grad=False, pretrained='imagenet')
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

    def _compute_metrics(
            self,
            trues_df: pd.DataFrame,
            preds: t.Dict,
            top_n_ranks: t.Tuple = (1, 5)
    ):
        true_labels = []
        pred_labels = []
        for name in preds.keys():
            true_labels.append(trues_df[trues_df['names'] == name]['labels'].values.tolist()[0])
            pred_labels.append(preds[name])
        for rank in top_n_ranks:
            top_rank_acc = self.top_n_accuracy(preds=pred_labels, truths=true_labels, n=rank)
            logging.info(f"\tTOP {rank} ACCURACY: {top_rank_acc * 100:.2f}"
                         f"\tTOP {rank} ERROR: {(1 - top_rank_acc) * 100:.2f}")

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

    def process(self, path_to_images: str):
        logging.info(f"\tBENCHMARK FOR {self.model_name}")
        labels_df = self._prepare_data(path_to_images=path_to_images)
        model = self._load_model()
        dataloader = self._make_dataloader(data_df=labels_df)
        preds_dict = self._inference_loop(dataloader=dataloader, model=model)
        self._compute_metrics(trues_df=labels_df, preds=preds_dict)


def parse_args():
    # Default args
    path_to_images = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
    model_name = "genet_small"
    in_size = 224
    batch_size = 128
    workers = 11

    parser = ArgumentParser()
    parser.add_argument('--path_to_images', default=path_to_images, type=str)
    parser.add_argument('--model_name', default=model_name, type=str)
    parser.add_argument('--in_size', default=in_size, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--workers', default=workers, type=int)
    return parser.parse_args()


def main(args):
    bench_obj = SimpleBenchmark(
        model_name=args.model_name,
        batch_size=args.batch_size,
        workers=args.workers,
        in_size=args.in_size
    )
    bench_obj.process(path_to_images=args.path_to_images)


if __name__ == '__main__':
    args = parse_args()
    main(args)
