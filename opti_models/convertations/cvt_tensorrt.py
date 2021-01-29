import os
import onnx
import typing as t
import tensorrt as trt
import argparse
import logging

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
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
        verbose: bool = False
) -> trt.ICudaEngine:

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(trt_logger) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, trt_logger) as parser:

        builder.max_workspace_size = 1 << 30

        if trt_engine_datatype == trt.DataType.HALF:
            builder.fp16_mode = True
        builder.max_batch_size = batch_size

        with open(uff_model_path, 'rb') as model:
            if verbose:
                logging.info(f"\t{sub_prefix}ONNX file parsing: START")
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logging.error(f"\t{sub_prefix}Error while parsing: {parser.get_error(error)}")
        if verbose:
            logging.info(f"\t{sub_prefix}Num of network layers: {network.num_layers}")
            logging.info(f"\t{sub_prefix}Building TensorRT engine. This may take a while...")

        return builder.build_cuda_engine(network)


def run_checks(precision: str, onnx_model_path: str):
    if not os.path.isfile(onnx_model_path):
        raise Exception('ONNX file does not exist. Check the path')
    if precision == '32':
        trt_datatype = trt.DataType.FLOAT
    elif precision == '16':
        trt_datatype = trt.DataType.HALF
    else:
        raise Exception('Wrong precision. Use either 32 of 16')

    return trt_datatype


def get_input_shape(model_path: str) -> t.Tuple:
    model = onnx.load(model_path)
    layer = model.graph.input[0]
    tensor_type = layer.type.tensor_type
    size = []
    if (tensor_type.HasField("shape")):
        for d in tensor_type.shape.dim:
            if (d.HasField("dim_value")):
                size.append(d.dim_value)
    return tuple(size)


def make_trt_convertation(
        precision: str,
        export_dir: str,
        onnx_model_path: str,
        verbose: bool = True
    ):

    if verbose:
        logging.info("\tConvert to TensorRT: START")

    bs, c, h, w = get_input_shape(model_path=onnx_model_path)

    model_name = onnx_model_path.split("/")[-2]
    export_dir = os.path.join(export_dir, model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    out_model_name = f"{model_name}_bs-{bs}_res-{c}x{h}x{w}.engine"
    export_path = os.path.join(export_dir, out_model_name)

    trt_datatype = run_checks(precision, onnx_model_path)
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
        trt_engine = build_engine(
            uff_model_path=onnx_model_path,
            trt_logger=TRT_LOGGER,
            trt_engine_datatype=trt_datatype,
            batch_size=bs,
            verbose=verbose
        )
    except Exception as e:
        logging.info("\tTensorRT engine build: FAIL")
        raise e
    else:
        if verbose:
            logging.info(f"\t{sub_prefix}TensorRT engine build: SUCCESS")
    # Save the engine to file

    try:
        save_engine(trt_engine, export_path)
    except:
        logging.info(f"\t{sub_prefix}TensorRT engine save: FAIL")
    else:
        if verbose:
            logging.info(f"\t{sub_prefix}TensorRT engine save: SUCCESS")
            logging.info("\tConvert to TensorRT: SUCCESS")


def main(args):
    onnx_model_path = args.onnx_path
    precision = args.precision
    export_dir = args.export_dir
    verbose = args.verbose

    make_trt_convertation(
        precision=precision,
        export_dir=export_dir,
        onnx_model_path=onnx_model_path,
        verbose=verbose
    )


def parse_args():
    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('--onnx-path', type=str, required=True)
    parser.add_argument('--export-dir', default='data/trt-export', type=str)
    parser.add_argument('--precision', default="32", type=str)
    parser.add_argument('--verbose', type=bool, default=True)

    return parser.parse_args()


def cvt_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones
    trt_export_path = "/mnt/Disk_F/Programming/pet_projects/opti_models/opti_models/convertations/data/trt-export"
    model_names = [name for name in show_available_backbones() if name not in os.listdir(trt_export_path)]
    onnx_models = "/mnt/Disk_F/Programming/pet_projects/opti_models/opti_models/convertations/data/onnx-export"
    for name in model_names:
        logging.info(f"\t{name.upper()} CONVERT")
        onnx_model_folder = os.path.join(onnx_models, name)
        onnx_model_name = [f for f in os.listdir(onnx_model_folder) if f.endswith("_simplified.onnx")][0]
        onnx_model_path = os.path.join(onnx_model_folder, onnx_model_name)
        args = parse_args()
        args.onnx_path = onnx_model_path
        main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)

    # cvt_all()
