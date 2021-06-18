#!/usr/bin/env python
import gc
import json
import logging
import os
import typing as t
from argparse import ArgumentParser
from time import perf_counter

import cv2
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import tensorrt as trt
from albumentations import Compose, Normalize, Resize
from tqdm import tqdm

from opti_models.utils.benchmarks_utils import (
    allocate_buffers,
    compute_metrics,
    get_shapes,
    load_engine,
    prepare_data,
)

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTBenchmark:
    def __init__(self, trt_path: str, export_name: str):
        """Class for simple TensorRT benchmarking

        Args:
            trt_path: Path to converted TensorRT model
        """
        self.model_name = trt_path.split("/")[-2]

        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = load_engine(trt_runtime=self.trt_runtime, engine_path=trt_path)
        self.context = self.trt_engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)

        bs, c, h, w = self.context.get_binding_shape(binding=0)
        self.max_batch_size = self.trt_engine.max_batch_size
        shapes = get_shapes(engine=self.trt_engine)
        self.num_classes = shapes[-1] // self.max_batch_size
        input_volume = trt.volume((c, w, h))
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))
        self.batch_size = bs

        self.augmentations = Compose(
            [
                Resize(height=h, width=w, interpolation=cv2.INTER_AREA),
                Normalize(),
            ],
            p=1,
        )

        self.export_name = export_name

    def _load_images(self, labels_df: pd.DataFrame, idx: int) -> t.Tuple[np.ndarray, t.List, int]:
        idx_start = idx
        idx_stop = min(idx + self.max_batch_size, labels_df.shape[0])
        bs = idx_stop - idx_start
        batch_df = labels_df.iloc[idx_start:idx_stop, :]
        images = np.asarray(
            [
                np.array(
                    self.augmentations(image=cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))['image'],
                    dtype=np.float32,
                    order='C',
                )
                .transpose((2, 0, 1))
                .ravel()
                for name in batch_df['names'].values
            ]
        )
        names = [name for name in batch_df['names'].values]
        return images, names, bs

    def _inference_loop(self, labels_df: pd.DataFrame) -> t.Dict:
        preds_dict = {}
        avg_batch_time = []
        for idx in tqdm(range(0, labels_df.shape[0], self.max_batch_size)):
            image_batch, names, bs = self._load_images(labels_df=labels_df, idx=idx)

            # TODO: remove this hack
            if bs != self.max_batch_size:
                continue

            start = perf_counter()
            np.copyto(self.inputs[0].host, image_batch.ravel())
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            self.context.execute_async(batch_size=bs, bindings=self.bindings, stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
            preds = [out.host for out in self.outputs][0]
            preds = np.asarray([preds[i : i + self.num_classes] for i in range(0, len(preds), self.num_classes)])
            end = perf_counter()

            avg_batch_time.append(end - start)
            preds_dict.update({name: label for name, label in zip(names, preds)})
            del preds
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

    def process(self, path_to_images: str, ranks: t.Tuple = (1, 5)) -> t.Dict:
        labels_df = prepare_data(path_to_images=path_to_images)

        logging.info(f"\tTENSORRT BENCHMARK FOR {self.model_name}: START")
        results_dict = self._inference_loop(labels_df=labels_df)
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

        # Clean
        self.context.__del__()
        self.trt_engine.__del__()
        del self.trt_engine, self.context, self.trt_runtime, self.inputs, self.outputs, self.bindings, self.stream
        gc.collect()

        return out_dict


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"
    parser = ArgumentParser(description='Simple speed benchmark, based on TRT models')
    parser.add_argument('--trt-path', required=True, type=str, help="Path to TRT model")
    parser.add_argument(
        '--export-name',
        required=False,
        type=str,
        default=None,
        help="File where to store bench statistics. If None, no statistics will be saved. Default: None",
    )
    parser.add_argument(
        '--path-to-images',
        required=False,
        type=str,
        default=path,
        help=f"Path to the validation images. Default: {path}",
    )
    return parser.parse_args()


def main(args) -> t.Dict:
    bench_obj = TensorRTBenchmark(trt_path=args.trt_path, export_name=args.export_name)
    statistics_dict = bench_obj.process(path_to_images=args.path_to_images)
    del bench_obj
    gc.collect()
    return statistics_dict


if __name__ == '__main__':
    args = parse_args()
    main(args)
