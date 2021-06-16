import argparse
import logging

from opti_models.benchmarks.imagenet_torch_benchmark import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"

    parser = argparse.ArgumentParser(description='Simple speed benchmark, based on pyTorch models')
    parser.add_argument('--model-name', type=str, help="Name of the model to test", default='resnet18')
    parser.add_argument(
        '--path-to-images', default=path, type=str, help=f"Path to the validation images, default: {path}"
    )
    parser.add_argument('--size', default=(224, 224), nargs='+', type=int, help="Input shape, default=(224, 224)")
    parser.add_argument('--batch-size', default=1, type=int, help="Size of the batch of images, default=1")
    parser.add_argument('--workers', default=1, type=int, help="Number of workers, default=1")
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
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)}: {model_name.upper()}")
        args = parse_args()
        args.model_name = model_name
        if model_name == "genet_large":
            args.in_size = (256, 256)
        elif model_name == 'inception_v3':
            args.in_size = (299, 299)

        try:
            main(args=args)
        except Exception as e:
            logging.info(f"\tCan't bench {model_name} \n{repr(e)}")
        logging.info(f"-" * 100)


if __name__ == '__main__':
    bench_all()
