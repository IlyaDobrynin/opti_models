import argparse
import logging
import os

from opti_models.benchmarks.imagenet_tensorrt_benchmark import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"
    parser = argparse.ArgumentParser(description='Simple speed benchmark, based on TRT models')
    parser.add_argument('--trt-path', type=str, help="Path to TRT model")
    parser.add_argument(
        '--path-to-images', default=path, type=str, help=f"Path to the validation images, default: {path}"
    )
    return parser.parse_args()


def bench_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones

    excluded_names = [
        'efficientnet_b1b',
        'efficientnet_b2b',
        'efficientnet_b3b',
        'efficientnet_b4b',
        'efficientnet_b5b',
        'efficientnet_b6b',
        'efficientnet_b7b',
        'efficientnet_b0c',
        'efficientnet_b1c',
        'efficientnet_b2c',
        'efficientnet_b3c',
        'efficientnet_b4c',
        'efficientnet_b5c',
        'efficientnet_b6c',
        'efficientnet_b7c',
        'efficientnet_b8c',
    ]
    model_names = [name for name in show_available_backbones() if name not in excluded_names]
    trt_models_path = "data/trt-export"
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)}")
        trt_model_path = os.path.join(trt_models_path, model_name)
        trt_model_names = [f for f in os.listdir(trt_model_path) if f.endswith(".engine")]
        for trt_model_name in trt_model_names:
            name_list = trt_model_name.split("_")
            for n in name_list:
                if n.startswith("prec"):
                    precision = n.split("-")[1]
                else:
                    raise ValueError(f"Can't find precision in trt_model_name: {trt_model_name}")
            logging.info(f"\tPRECISION: {precision} -------------")
            trt_path = os.path.join(trt_model_path, trt_model_name)
            args = parse_args()
            args.trt_path = trt_path
            main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    bench_all()
