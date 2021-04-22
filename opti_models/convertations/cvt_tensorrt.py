#!/usr/bin/env python
import argparse
import logging
import os
import typing as t

import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
sub_prefix = ">>>>> "

__all__ = ["make_trt_convertation"]

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch.transforms import ToTensorV2
import sys, os, argparse, logging, random, cv2, glob

def preprocess(image, size):
    transforms = Compose([
            Resize(
                height=size[0],
                width=size[1],
                interpolation=cv2.INTER_AREA
            ),
            Normalize(),
            ToTensorV2(),
        ], p=1)

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = transforms(image=img)["image"]

    return input_data


def load_data(data_dir, onnx_model_path, num_calibration_images):
    bs, c, h, w = get_input_shape(model_path=onnx_model_path)
    images = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    num_batches = num_calibration_images + (num_calibration_images % 1 > 0)
    print("Num batches: ", num_batches)
    img = 0
    batches = np.zeros(shape=(num_batches,c,h,w),dtype=np.float32)
    for i in range(num_batches-1):
        batches[i] = preprocess(images[img], (w, h))
        img += 1
    return batches

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, calibration_images, batch_size, onnx_model_path, num_calibration_images):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = load_data(calibration_images, onnx_model_path, num_calibration_images)
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, _):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None

        batch = self.data[self.current_index:self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def save_engine(engine: trt.ICudaEngine, engine_dest_path: str):
    logging.info(f"\t{sub_prefix}Saving TensorRT engine")
    buf = engine.serialize()
    with open(engine_dest_path, 'wb') as f:
        f.write(buf)


def build_engine(
    uff_model_path: str,
    trt_logger: trt.Logger,
    trt_engine_datatype: trt.DataType = trt.DataType.FLOAT,
    batch_size: int = 1,
    verbose: bool = False,
    calibration_images=None,
    onnx_model_path=None,
    num_calibration_images=None

) -> trt.ICudaEngine:

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
        network, trt_logger
    ) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size

        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        elif trt_engine_datatype == trt.DataType.INT8:
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = Int8EntropyCalibrator(cache_file="calibration.cache", calibration_images=calibration_images,
                                                           batch_size=batch_size, onnx_model_path=onnx_model_path,
                                                           num_calibration_images=num_calibration_images)

        with open(uff_model_path, 'rb') as model:
            if verbose:
                logging.info(f"\t{sub_prefix}ONNX file parsing: START")
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logging.error(f"\t{sub_prefix}Error while parsing: {parser.get_error(error)}")
        if verbose:
            logging.info(f"\t{sub_prefix}Num of network layers: {network.num_layers}")
            logging.info(f"\t{sub_prefix}Building TensorRT engine. This may take a while...")

        return builder.build_engine(network, config)


def run_checks(precision: str, onnx_model_path: str, calibration_images:str, num_calibration_images:int):
    if not os.path.isfile(onnx_model_path):
        raise Exception('ONNX file does not exist. Check the path')
    if precision == '32':
        trt_datatype = trt.DataType.FLOAT
    elif precision == '16':
        trt_datatype = trt.DataType.HALF
    elif precision == '8':
        trt_datatype = trt.DataType.INT8
        if (not calibration_images) or (not num_calibration_images):
            raise Exception('For INT8 convertation you need to supply calibration_images directory, and number of calibration images')
    else:
        raise Exception('Wrong precision. Use either 32, 16, or 8')

    return trt_datatype


def get_input_shape(model_path: str) -> t.Tuple:
    model = onnx.load(model_path)
    layer = model.graph.input[0]
    tensor_type = layer.type.tensor_type
    size = []
    if tensor_type.HasField("shape"):
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                size.append(d.dim_value)
    return tuple(size)


def make_trt_convertation(precision: str, export_name: str, onnx_model_path: str, verbose: bool = True,
                          calibration_images=None, num_calibration_images=None):

    if verbose:
        logging.info("\tConvert to TensorRT: START")

    bs, c, h, w = get_input_shape(model_path=onnx_model_path)

    model_name = onnx_model_path.split("/")[-2]
    export_dir = os.path.join('data/trt-export', model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    if not export_name:
        out_model_name = f"{model_name}_prec-{precision}_bs-{bs}_res-{c}x{h}x{w}.engine"
    else:
        out_model_name = f"{export_name}.engine"
    export_path = os.path.join(export_dir, out_model_name)

    trt_datatype = run_checks(precision, onnx_model_path, calibration_images, num_calibration_images)
    # We first load all custom plugins shipped with TensorRT,
    # some of them will be needed during inference
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    if verbose:
        # Display requested engine settings to stdout
        logging.info(f"\t{sub_prefix}TensorRT inference engine settings:")
        logging.info(f"\t{sub_prefix}  * Inference precision - {trt_datatype}")
        logging.info(f"\t{sub_prefix}  * Max batch size - {bs}")

    # This function uses supplied .uff file
    # alongside with UffParser to build TensorRT
    # engine. For more details, check implementation
    try:
        if trt_datatype != trt.DataType.INT8:
            params = {'uff_model_path': onnx_model_path,
                    'trt_logger': TRT_LOGGER,
                    'trt_engine_datatype': trt_datatype,
                    'batch_size': bs,
                    'verbose': verbose}
        else:
            params = {'uff_model_path': onnx_model_path,
                      'trt_logger': TRT_LOGGER,
                      'trt_engine_datatype': trt_datatype,
                      'batch_size': bs,
                      'verbose': verbose,
                      'calibration_images': calibration_images,
                      'onnx_model_path': onnx_model_path,
                      'num_calibration_images': num_calibration_images
                      }

        trt_engine = build_engine(**params)
    except Exception as e:
        logging.info("\tTensorRT engine build: FAIL")
        raise e
    else:
        if verbose:
            logging.info(f"\t{sub_prefix}TensorRT engine build: SUCCESS")

    # Save the engine to file
    try:
        save_engine(trt_engine, export_path)
    except Exception as e:
        logging.info(f"\t{sub_prefix}TensorRT engine save: FAIL")
        raise e
    else:
        if verbose:
            logging.info(f"\t{sub_prefix}TensorRT engine save: SUCCESS")
            logging.info("\tConvert to TensorRT: SUCCESS")


def main(args):
    onnx_model_path = args.onnx_path
    precision = args.precision
    export_name = args.export_name
    verbose = args.verbose
    calibration_images = args.calibration_images
    num_calibration_images = args.num_calibration_images

    make_trt_convertation(
        precision=precision, export_name=export_name, onnx_model_path=onnx_model_path, verbose=verbose,
        calibration_images=calibration_images, num_calibration_images=num_calibration_images
    )


def parse_args():
    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('--onnx-path', type=str, required=True)
    parser.add_argument('--export-name', type=str)
    parser.add_argument('--precision', default="32", type=str)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--calibration-images', type=str, default=None)
    parser.add_argument('--num-calibration-images', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args=args)

    # cvt_all()

