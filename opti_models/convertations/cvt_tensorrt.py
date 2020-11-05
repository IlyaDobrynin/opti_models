import os
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
        info: bool = False
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
            if info:
                logging.info(f"\t{sub_prefix}ONNX file parsing: START")
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logging.error(f"\t{sub_prefix}Error while parsing: {parser.get_error(error)}")
        if info:
            logging.info(f"\t{sub_prefix}Num of network layers: {network.num_layers}")
            logging.info(f"\t{sub_prefix}Building TensorRT engine. This may take a while...")

        return builder.build_cuda_engine(network)


def run_checks(FP: str, onnx_model_path: str):
    if not os.path.isfile(onnx_model_path):
        raise Exception('ONNX file does not exist. Check the path')
    if FP == '32':
        trt_datatype = trt.DataType.FLOAT
    elif FP == '16':
        trt_datatype = trt.DataType.HALF
    else:
        raise Exception('Wrong precision. Use either 32 of 16')

    return trt_datatype


def make_trt_convertation(
        fp_type: str,
        export_dir: str,
        onnx_model_path: str,
        batch_size: int,
        in_size: t.Tuple,
        info: bool = True
):
    if info:
        logging.info("\tConvert to TensorRT: START")

    model_name = onnx_model_path.split("/")[-2]

    export_dir = os.path.join(export_dir, model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    out_model_name = f"{model_name}_bs-{batch_size}_res-{in_size[0]}x{in_size[0]}.engine"
    export_path = os.path.join(export_dir, out_model_name)

    trt_datatype = run_checks(fp_type, onnx_model_path)
    # We first load all custom plugins shipped with TensorRT,
    # some of them will be needed during inference
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    if info:
        # Display requested engine settings to stdout
        logging.info(f"\t{sub_prefix}TensorRT inference engine settings:")
        logging.info(f"\t{sub_prefix}  * Inference precision - {trt_datatype}")
        logging.info(f"\t{sub_prefix}  * Max batch size - {batch_size}")

    # This function uses supplied .uff file
    # alongside with UffParser to build TensorRT
    # engine. For more details, check implmentation
    trt_engine = build_engine(
        uff_model_path=onnx_model_path,
        trt_logger=TRT_LOGGER,
        trt_engine_datatype=trt_datatype,
        batch_size=batch_size,
        info=info
    )
    if info:
        logging.info(f"\t{sub_prefix}TensorRT engine build: SUCCESS")
    # Save the engine to file
    save_engine(trt_engine, export_path)

    if info:
        logging.info(f"\t{sub_prefix}TensorRT engine save: SUCCESS")
        logging.info("\tConvert to TensorRT: SUCCESS")


def main(args):
    onnx_model_path = args.onnx_path
    batch_size = args.batch_size
    in_size = args.in_size
    fp_type = args.fp_type
    export_dir = args.export_dir
    info = args.info

    make_trt_convertation(
        fp_type=fp_type,
        export_dir=export_dir,
        onnx_model_path=onnx_model_path,
        batch_size=batch_size,
        in_size=in_size,
        info=info
    )


def parse_args():
    onnx_path = ""
    export_dir = "../../data/trt_export"
    in_size = (224, 224)

    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('--onnx_path', default=onnx_path, type=str)
    parser.add_argument('--export_dir', default=export_dir, type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--in_size', nargs="+", default=in_size, type=int)
    parser.add_argument('--fp_type', default="32", type=str)
    parser.add_argument('--info', type=bool, default=True)

    return parser.parse_args()


def cvt_all():
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

    onnx_models = "../../data/onnx_export"
    for name in model_names:
        logging.info(f"{name.upper()} CONVERT")
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
