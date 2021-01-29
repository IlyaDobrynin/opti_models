import os
import typing as t
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import logging
import tensorrt as trt
import pycuda.driver as cuda
from argparse import ArgumentParser
from albumentations import Compose, Resize, Normalize
import cv2
from opti_models.utils.benchmarks_utils import compute_metrics, prepare_data, load_engine, allocate_buffers, get_shapes

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTBenchmark:
    def __init__(self, trt_path: str):
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

        self.augmentations = Compose([
            Resize(
                height=h,
                width=w,
                interpolation=cv2.INTER_AREA
            ),
            Normalize(),
        ], p=1)

    def _load_images(self, labels_df: pd.DataFrame, idx: int):
        idx_start = idx
        idx_stop = min(idx + self.max_batch_size, labels_df.shape[0])
        bs = idx_stop - idx_start
        batch_df = labels_df.iloc[idx_start: idx_stop, :]
        images = np.asarray([
            np.array(
                self.augmentations(image=cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))['image'],
                dtype=np.float32,
                order='C'
            ).transpose((2, 0, 1)).ravel() for name in batch_df['names'].values
        ])
        names = [name for name in batch_df['names'].values]
        return images, names, bs

    def _inference_loop(self, labels_df: pd.DataFrame):
        preds_dict = {}
        avg_batch_time = []
        for idx in tqdm(range(0, labels_df.shape[0], self.max_batch_size)):
            image_batch, names, bs = self._load_images(labels_df=labels_df, idx=idx)

            # TODO: remove this hack
            if bs != self.max_batch_size:
                continue

            batch_time = time()
            np.copyto(self.inputs[0].host, image_batch.ravel())
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
            self.context.execute_async(
                batch_size=bs,
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
            preds = [out.host for out in self.outputs][0]
            preds = np.asarray([preds[i: i + self.num_classes] for i in range(0, len(preds), self.num_classes)])
            avg_batch_time.append(time() - batch_time)
            preds_dict.update({name: label for name, label in zip(names, preds)})
        logging.info(f"\tAverage fps: {self.batch_size / np.mean(avg_batch_time)}")
        return preds_dict

    def process(self, path_to_images: str, ranks: t.Tuple = (1, 5)):
        labels_df = prepare_data(path_to_images=path_to_images)

        logging.info(f"\tTENSORRT BENCHMARK FOR {self.model_name}: START")
        preds_dict = self._inference_loop(labels_df=labels_df)
        rank_metrics = compute_metrics(trues_df=labels_df, preds=preds_dict, top_n_ranks=ranks)
        for rank, rank_metric in zip(ranks, rank_metrics):
            logging.info(f"\tTOP {rank} ACCURACY: {rank_metric * 100:.2f}"
                         f"\tTOP {rank} ERROR: {(1 - rank_metric) * 100:.2f}")
        logging.info(f"\tBENCHMARK FOR {self.model_name}: SUCCESS")


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"
    parser = ArgumentParser(description='Simple speed benchmark, based on TRT models')
    parser.add_argument('--trt-path', type=str, help="Path to TRT model", required=True)
    parser.add_argument('--path-to-images', default=path, type=str, help=f"Path to the validation images, default: {path}")
    return parser.parse_args()


def main(args):
    bench_obj = TensorRTBenchmark(trt_path=args.trt_path)
    bench_obj.process(path_to_images=args.path_to_images)


def bench_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones
    model_names = show_available_backbones()
    trt_models_path = "/mnt/Disk_F/Programming/pet_projects/opti_models/opti_models/convertations/data/trt-export"
    for name in model_names:
        trt_model_path = os.path.join(trt_models_path, name)
        trt_model_name = [f for f in os.listdir(trt_model_path) if f.endswith(".engine")][0]
        trt_path = os.path.join(trt_model_path, trt_model_name)
        args = parse_args()
        args.trt_path = trt_path
        main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    args = parse_args()
    main(args)

    # bench_all()
