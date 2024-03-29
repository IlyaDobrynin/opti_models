import argparse
import logging
import os

from opti_models.convertations.cvt_onnx import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX conversion script')
    parser.add_argument('--model-name', type=str, help="Name of the model")
    parser.add_argument('--model-type', default='classifier', type=str, help="Type of the model")
    parser.add_argument(
        '--model-path',
        type=str,
        default='ImageNet',
        help="Path to the pretrained weights, or ImageNet, if you want to get model with imagenet pretrain",
    )
    parser.add_argument('--export-dir', type=str, default='data/onnx-export', help="Path to directory to save results")
    parser.add_argument('--export-name', type=str, required=False, help="Name of the exported onnx file")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for optimized model")
    parser.add_argument('--size', nargs='+', default=(3, 224, 224), type=int, help="Size of the input tensor")
    parser.add_argument('--num-classes', default=1000, type=int, help="Number of output classes of the model")
    parser.add_argument('--verbose', default=True, type=bool, help="Flag for showing information")

    return parser.parse_args()


def cvt_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones

    included_names = [name for name in show_available_backbones()]
    excluded_names = []
    onnx_export = 'data/onnx-export'
    if not os.path.exists(onnx_export):
        os.makedirs(onnx_export, exist_ok=True)
    model_names = [
        name for name in included_names if (name not in excluded_names) and (name not in os.listdir(onnx_export))
    ]
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)} - {model_name.upper()} CONVERT")
        args = parse_args()
        args.model_name = model_name
        if model_name == "genet_large":
            args.size = (3, 256, 256)
        elif model_name == 'inception_v3':
            args.size = (3, 299, 299)

        try:
            main(args=args)
            logging.info(f"-" * 100)
        except Exception as e:
            logging.info(f"\tCan't convert model: {model_name}\n{repr(e)}")


if __name__ == '__main__':
    cvt_all()
