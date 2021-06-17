#!/usr/bin/env python
import argparse
import gc
import logging
import os
import typing as t

import tensorrt as trt

from opti_models.utils.convertations_utils import Int8EntropyCalibrator, get_input_shape

logging.basicConfig(level=logging.INFO)
sub_prefix = ">>>>> "

__all__ = ["make_trt_convertation"]


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
    calibration_images_dir: str = None,
    preprocess_method: callable = None,
    export_dir: str = None,
) -> trt.ICudaEngine:

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
        network, trt_logger
    ) as parser:

        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size

        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        elif trt_engine_datatype == trt.DataType.INT8:
            builder.int8_mode = True
            builder.int8_calibrator = Int8EntropyCalibrator(
                cache_file=os.path.join(export_dir, "int8_calibration.cache"),
                calibration_images_dir=calibration_images_dir,
                batch_size=batch_size,
                onnx_model_path=uff_model_path,
                preprocess_method=preprocess_method,
            )

        with open(uff_model_path, 'rb') as model:
            if verbose:
                logging.info(f"\t{sub_prefix}ONNX file parsing: START")
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logging.error(f"\t{sub_prefix}Error while parsing: {parser.get_error(error)}")
        if verbose:
            logging.info(f"\t{sub_prefix}Num of network layers: {network.num_layers}")
            logging.info(f"\t{sub_prefix}Building TensorRT engine. This may take a while...")

        engine = builder.build_cuda_engine(network)

        return engine


def run_checks(precision: str, onnx_model_path: str, calibration_images_dir: str) -> trt.DataType:
    if not os.path.isfile(onnx_model_path):
        raise Exception('ONNX file does not exist. Check the path')
    if precision == '32':
        trt_datatype = trt.DataType.FLOAT
    elif precision == '16':
        trt_datatype = trt.DataType.HALF
    elif precision == '8':
        trt_datatype = trt.DataType.INT8
        if (calibration_images_dir is None) or (not os.path.exists(calibration_images_dir)):
            raise Exception('For INT8 convertation you need to provide calibration_images directory')
    else:
        raise Exception('Wrong precision. Use either 32, 16, or 8')

    return trt_datatype


def make_trt_convertation(
    precision: str,
    export_name: str,
    onnx_model_path: str,
    verbose: bool = True,
    calibration_images_dir: str = None,
    preprocess_method: callable = None,
):
    if verbose:
        logging.info("\tConvert to TensorRT: START")

    bs, c, h, w = get_input_shape(model_path=onnx_model_path)

    model_name = onnx_model_path.split("/")[-2]
    export_dir = os.path.join('data/trt-export', model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    if export_name is None:
        out_model_name = f"{model_name}_prec-{precision}_bs-{bs}_res-{c}x{h}x{w}.engine"
    else:
        out_model_name = f"{export_name}.engine"
    export_path = os.path.join(export_dir, out_model_name)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    trt_datatype = run_checks(
        precision=precision, onnx_model_path=onnx_model_path, calibration_images_dir=calibration_images_dir
    )
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
            params = {
                'uff_model_path': onnx_model_path,
                'trt_logger': TRT_LOGGER,
                'trt_engine_datatype': trt_datatype,
                'batch_size': bs,
                'verbose': verbose,
            }
        else:
            params = {
                'uff_model_path': onnx_model_path,
                'trt_logger': TRT_LOGGER,
                'trt_engine_datatype': trt_datatype,
                'batch_size': bs,
                'verbose': verbose,
                'calibration_images_dir': calibration_images_dir,
                'preprocess_method': preprocess_method,
                'export_dir': export_dir,
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

    # Clean
    trt_engine.__del__()
    del TRT_LOGGER, trt_engine
    gc.collect()


def main(args):
    onnx_model_path = args.onnx_path
    precision = args.precision
    export_name = args.export_name
    verbose = args.verbose
    calibration_images_dir = args.calibration_images_dir

    make_trt_convertation(
        precision=precision,
        export_name=export_name,
        onnx_model_path=onnx_model_path,
        verbose=verbose,
        calibration_images_dir=calibration_images_dir,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='TRT conversion parameters')
    parser.add_argument('--onnx-path', type=str, required=True, help='Path to ONNX file')
    parser.add_argument(
        '--export-name',
        type=str,
        required=False,
        default=None,
        help="Output name. Default {model_name}_prec-{precision}_bs-{bs}_res-{c}x{h}x{w}.engine",
    )
    parser.add_argument(
        '--precision',
        required=False,
        default="32",
        type=str,
        help="Precision for the trt model: 32, 16 or 8. Default 32.",
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True, required=False, help="Flag to show out info. Default True."
    )
    parser.add_argument(
        '--calibration-images-dir',
        type=str,
        default=None,
        required=False,
        help="Path to calibraton images directory (required for precision=8). Default None.",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
