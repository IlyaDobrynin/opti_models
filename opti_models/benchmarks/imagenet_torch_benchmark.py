#!/usr/bin/env python
import json
import logging
import os
import typing as t
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from opti_models.benchmarks.datasets import ImagenetDataset
from opti_models.models.backbones.backbone_factory import show_available_backbones
from opti_models.utils.benchmarks_utils import compute_metrics, prepare_data
from opti_models.utils.model_utils import get_model

logging.basicConfig(level=logging.INFO)


class TorchBenchmark:
    def __init__(
        self,
        model_name: str,
        export_name: str = None,
        batch_size: t.Optional[int] = 1,
        in_size: t.Tuple = (224, 224),
        workers: t.Optional[int] = 1,
        show_model_info: bool = False,
    ):
        """Class for simple torch benchmarking

        Args:
            model_name: Name of the model to bench
            batch_size: Batch size
            in_size: Image input size
            workers: Number of workers in dataloader
            show_model_info: Flag to chow model info
        """
        self.model_name = model_name
        self.export_name = export_name
        self.batch_size = batch_size
        self.workers = workers
        self.in_size = in_size
        self.show_model_info = show_model_info

    def _make_dataloader(self, data_df: pd.DataFrame) -> DataLoader:
        dataset_obj = ImagenetDataset(data_df=data_df, in_size=self.in_size)
        dataloader = DataLoader(
            dataset=dataset_obj,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=dataset_obj.collate_fn,
            pin_memory=True,
        )
        return dataloader

    @staticmethod
    def _prediction_step(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        preds = model(inputs)
        preds = F.softmax(preds, dim=-1)
        return preds

    def _inference_loop(self, dataloader: DataLoader, model: torch.nn.Module) -> t.Dict:
        preds_dict = {}
        avg_batch_time = []

        torch.cuda.synchronize()
        model.eval()
        for batch in tqdm(dataloader, total=len(dataloader)):
            inputs = batch[0].cuda()
            names = batch[2]
            start = perf_counter()
            preds = self._prediction_step(model=model, inputs=inputs)
            torch.cuda.synchronize()
            end = perf_counter()
            time = end - start
            avg_batch_time.append(time)

            preds = preds.data.cpu().numpy()
            preds_dict.update({name: label for name, label in zip(names, preds)})

        ips = self.batch_size / np.mean(avg_batch_time)
        img_time = np.mean(avg_batch_time) * 1000
        logging.info(f"\tAverage images per second: {ips:.4f} image/s")
        logging.info(f"\tAverage second for image: {img_time:.4f} ms")

        out_dict = {'predictions': preds_dict, 'ips': ips, 'img_time': img_time}

        return out_dict

    def _save_statistics(self, out_dict: t.Dict):
        export_path = os.path.dirname(self.export_name)
        if not os.path.exists(export_path):
            os.makedirs(export_path, exist_ok=True)
        with open(self.export_name, 'w') as f:
            json.dump(out_dict, f)

    def process(self, path_to_images: str, ranks: t.Tuple = (1, 5)):
        labels_df = prepare_data(path_to_images=path_to_images)
        model = get_model(
            model_type='classifier', model_name=self.model_name, model_path='ImageNet', show=self.show_model_info
        )
        dataloader = self._make_dataloader(data_df=labels_df)

        logging.info(f"\tTORCH BENCHMARK FOR {self.model_name}: START")
        results_dict = self._inference_loop(dataloader=dataloader, model=model)
        preds_dict = results_dict['predictions']
        rank_metrics = compute_metrics(trues_df=labels_df, preds=preds_dict, top_n_ranks=ranks)

        out_dict = {'ips': results_dict['ips'], 'img_time': results_dict['img_time']}
        for rank, rank_metric in zip(ranks, rank_metrics):
            out_dict[f"top_{rank}_acc"] = rank_metric * 100
            out_dict[f"top_{rank}_err"] = (1 - rank_metric) * 100
            logging.info(
                f"\tTOP {rank} ACCURACY: {out_dict[f'top_{rank}_acc']:.2f}"
                f"\tTOP {rank} ERROR: {out_dict[f'top_{rank}_err']:.2f}"
            )
        logging.info(f"\tBENCHMARK FOR {self.model_name}: SUCCESS")

        # Save statistics
        if self.export_name is not None:
            self._save_statistics(out_dict=out_dict)
            logging.info(f"\tBENCHMARK STATS WERE SAVED AT {self.export_name}")


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"

    parser = ArgumentParser(description='Simple speed benchmark, based on pyTorch models')
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help=f"Name of the model to test. Available: {show_available_backbones()}",
    )
    parser.add_argument(
        '--export-name',
        type=str,
        required=False,
        default=None,
        help="File where to store bench statistics. If None, no statistics will be saved. Default: None",
    )
    parser.add_argument(
        '--path-to-images',
        default=path,
        required=False,
        type=str,
        help=f"Path to the validation images. Default: {path}.",
    )
    parser.add_argument(
        '--size', default=(224, 224), required=False, nargs='+', type=int, help="Input shape. Default = (224, 224)."
    )
    parser.add_argument(
        '--batch-size', default=1, required=False, type=int, help="Size of the batch of images. Default = 1."
    )
    parser.add_argument('--workers', default=1, required=False, type=int, help="Number of workers. Default = 1.")
    return parser.parse_args()


def main(args):
    bench_obj = TorchBenchmark(
        model_name=args.model_name,
        export_name=args.export_name,
        batch_size=args.batch_size,
        workers=args.workers,
        in_size=args.size,
    )
    bench_obj.process(path_to_images=args.path_to_images)
    del bench_obj
    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    main(args)
