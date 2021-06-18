import argparse
import logging
import os

from opti_models.convertations.cvt_tensorrt import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('--onnx-path', type=str)
    parser.add_argument('--export-dir', default='data/trt-export', type=str)
    parser.add_argument('--export-name', default=None, type=str)
    parser.add_argument('--precision', default="32", type=str)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--calibration-images-dir', type=str, default='/usr/local/opti_models/calibration_images')
    return parser.parse_args()


def cvt_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones

    trt_dir = "data/trt-export"
    onnx_models = "data/onnx-export"

    included_names = [name for name in show_available_backbones()]
    excluded_names = []
    model_names = [
        name
        for name in included_names
        if (name not in excluded_names) and (name in os.listdir(onnx_models)) and (name not in os.listdir(trt_dir))
    ]
    onnx_models = "data/onnx-export"
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)} - {model_name.upper()} CONVERT")
        precisions = ['32', '16', '8']
        for precision in precisions:
            logging.info(f"\tPRECISION: {precision} -------------")
            onnx_model_folder = os.path.join(onnx_models, model_name)
            onnx_model_names = [f for f in os.listdir(onnx_model_folder) if f.endswith("_simplified.onnx")]
            if len(onnx_model_names) > 0:
                onnx_model_name = onnx_model_names[0]
            else:
                continue

            onnx_model_path = os.path.join(onnx_model_folder, onnx_model_name)
            args = parse_args()
            args.onnx_path = onnx_model_path
            args.precision = precision
            try:
                main(args=args)
            except Exception as e:
                logging.info(f"\tCan't convert {model_name}\n{repr(e)}")
        logging.info(f"-" * 100)


if __name__ == '__main__':
    cvt_all()
