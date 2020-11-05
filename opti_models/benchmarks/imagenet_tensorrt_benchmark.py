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
from opti_models.utils.benchmarks_utils import compute_metrics, prepare_data, load_engine, allocate_buffers

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TensorRTBenchmark:
    def __init__(
            self,
            trt_path: str,
            in_size: t.Tuple = (224, 224)
    ):
        self.model_width = in_size[1]
        self.model_height = in_size[0]
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = load_engine(
            trt_runtime=self.trt_runtime,
            engine_path=trt_path
        )
        self.model_name = trt_path.split("/")[-2]
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)
        self.context = self.trt_engine.create_execution_context()
        input_volume = trt.volume((3, self.model_width, self.model_height))
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))
        self.in_size = in_size
        self.batch_size = 1

        self.augmentations = Compose([
            Resize(
                height=in_size[0],
                width=in_size[1],
                interpolation=cv2.INTER_AREA
            ),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ], p=1)

    def _inference_loop(self, labels_df: pd.DataFrame):
        preds_dict = {}
        avg_batch_time = []
        for idx, row in tqdm(labels_df.iterrows(), total=labels_df.shape[0]):
            name = row["names"]
            image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

            input_data = self.augmentations(image=image)["image"]
            input_data = np.array(input_data, dtype=np.float32, order='C')
            input_data = input_data.transpose((2, 0, 1))
            np.copyto(self.inputs[0].host, input_data.ravel())
            [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

            batch_time = time()
            self.context.execute_async(
                batch_size=self.batch_size,
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )
            avg_batch_time.append(time() - batch_time)

            [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
            self.stream.synchronize()
            pred = np.asarray([out.host for out in self.outputs])
            preds_dict.update({name: label for name, label in zip([name], pred)})
        logging.info(f"\tAverage fps: {self.batch_size / np.mean(avg_batch_time)}")
        return preds_dict

    def process(self, path_to_images: str, ranks: t.Tuple = (1, 5)):
        labels_df = prepare_data(path_to_images=path_to_images)

        logging.info(f"\tBENCHMARK FOR {self.model_name}: START")
        preds_dict = self._inference_loop(labels_df=labels_df)
        rank_metrics = compute_metrics(trues_df=labels_df, preds=preds_dict, top_n_ranks=ranks)
        for rank, rank_metric in zip(ranks, rank_metrics):
            logging.info(f"\tTOP {rank} ACCURACY: {rank_metric * 100:.2f}"
                         f"\tTOP {rank} ERROR: {(1 - rank_metric) * 100:.2f}")
        logging.info(f"\tBENCHMARK FOR {self.model_name}: SUCCESS")


def parse_args():
    # Default args
    path_to_images = "/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
    trt_path = "../../data/trt_export/resnet18/resnet18_bs-1_res-224x224.engine"
    in_size = (224, 224)

    parser = ArgumentParser()
    parser.add_argument('--path_to_images', default=path_to_images, type=str)
    parser.add_argument('--trt_path', default=trt_path, type=str)
    parser.add_argument('--in_size', default=in_size, nargs='+', type=int)
    return parser.parse_args()


def main(args):
    bench_obj = TensorRTBenchmark(
        trt_path=args.trt_path,
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
    trt_models_path = "../../data/trt_export"

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
