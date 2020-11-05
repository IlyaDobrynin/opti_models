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
from opti_models.utils.benchmarks_utils import compute_metrics, prepare_data
logging.basicConfig(level=logging.INFO)


class SimpleBenchmark:
    def __init__(
            self,
            model_name: str,
            batch_size: int,
            workers: int,
            in_size: t.Tuple = (224, 224)
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.workers = workers
        self.in_size = in_size

    def _load_model(self, show: bool = False):
        models_facade_obj = models_facade.ModelFacade(task="backbones")
        model = models_facade_obj.get_model_class(
            model_definition=self.model_name
        )(requires_grad=False, pretrained='imagenet')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(device)
        if show:
            summary(model, input_size=[3] + self.in_size)
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

    def _inference_loop(self, dataloader: DataLoader, model: torch.nn.Module):
        preds_dict = {}
        avg_batch_time = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].cuda()
            names = batch[2]

            batch_time = time()
            preds = model(inputs)
            avg_batch_time.append(time() - batch_time)

            preds = F.softmax(preds, dim=-1).data.cpu().numpy()
            preds_dict.update({name: label for name, label in zip(names, preds)})
        logging.info(f"\tAverage fps: {self.batch_size / np.mean(avg_batch_time)}")
        return preds_dict

    def process(self, path_to_images: str, ranks: t.Tuple = (1, 5)):
        labels_df = prepare_data(path_to_images=path_to_images)
        model = self._load_model()
        dataloader = self._make_dataloader(data_df=labels_df)

        logging.info(f"\tBENCHMARK FOR {self.model_name}")
        preds_dict = self._inference_loop(dataloader=dataloader, model=model)

        rank_metrics = compute_metrics(trues_df=labels_df, preds=preds_dict, top_n_ranks=ranks)
        for rank, rank_metric in zip(ranks, rank_metrics):
            logging.info(f"\tTOP {rank} ACCURACY: {rank_metric * 100:.2f}"
                         f"\tTOP {rank} ERROR: {(1 - rank_metric) * 100:.2f}")


def parse_args():
    # Default args
    path_to_images = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
    model_name = "genet_small"
    in_size = (224, 224)
    batch_size = 1
    workers = 1

    parser = ArgumentParser()
    parser.add_argument('--path_to_images', default=path_to_images, type=str)
    parser.add_argument('--model_name', default=model_name, type=str)
    parser.add_argument('--in_size', default=in_size, nargs='+', type=int)
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


def bench_all():
    model_names = [
        "resnet50",
        "resnet34",
        "resnet18",
        "mobilenetv2_w1",
        "mobilenetv2_wd2",
        "mobilenetv2_wd4",
        "mobilenetv2_w3d4",
        "mobilenetv3_large_w1",
        "mixnet_s",
        "mixnet_m",
        "mixnet_l",
        'efficientnet_b0',
        'efficientnet_b1',
        'genet_small',
        'genet_normal',
        'genet_large'
    ]

    for model_name in model_names:
        args = parse_args()
        args.model_name = model_name
        if model_name == "genet_large":
            args.in_size = (256, 256)
        main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # bench_all()