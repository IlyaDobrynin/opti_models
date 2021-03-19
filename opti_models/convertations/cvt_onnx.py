#!/usr/bin/env python
import argparse
import logging
import os
import typing as t

import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify

from opti_models.utils.model_utils import get_model

logging.basicConfig(level=logging.INFO)
sub_prefix = ">>>>> "

__all__ = ["make_onnx_convertation"]


def get_parameters(export_name: str, model_name: str, batch_size: int, in_size: t.Tuple) -> t.Tuple:
    export_dir = os.path.join('data/onnx-export', model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    if not export_name:
        out_model_name = f"{model_name}_bs-{batch_size}_res-{in_size[0]}x{in_size[1]}x{in_size[2]}"
    else:
        out_model_name = export_name
    export_path = os.path.join(export_dir, f"{out_model_name}.onnx")
    export_simple_path = os.path.join(export_dir, f"{out_model_name}_simplified.onnx")
    input_size = (batch_size, in_size[0], in_size[1], in_size[2])
    return export_path, export_simple_path, input_size


def make_onnx_convertation(
    export_name: str,
    batch_size: int,
    model_name: str,
    in_size: t.Tuple,
    model_type: str,
    num_classes: t.Optional[int] = 1000,
    model_path: t.Optional[str] = None,
    model: t.Optional[torch.nn.Module] = None,
    verbose: t.Optional[bool] = False,
):
    if verbose:
        logging.info("\tConvert to ONNX: START")

    export_path, export_simple_path, input_size = get_parameters(
        export_name=export_name, model_name=model_name, batch_size=batch_size, in_size=in_size
    )
    model = get_model(
        model_name=model_name,
        input_shape=input_size,
        model_type=model_type,
        model=model,
        num_classes=num_classes,
        model_path=model_path,
    )
    model.eval()
    dummy_input = torch.ones(input_size, device="cuda")
    with torch.no_grad():
        pre_det_res = model(dummy_input.cuda())
    dummy_output = torch.ones(*pre_det_res.shape, device="cuda")

    input_names = ["input"]
    output_names = ["output"]

    try:
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=export_path,
            input_names=input_names,
            output_names=output_names,
            example_outputs=dummy_output,
            opset_version=11,
            export_params=True,
        )
    except Exception as e:
        logging.info("\tConvert to ONNX: FAIL")
        raise e
    else:
        if verbose:
            logging.info("\tConvert to ONNX: SUCCESS")

    try:
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        logging.info("\tConvert to ONNX: FAIL")
        raise e
    else:
        if verbose:
            logging.info("\tONNX check: SUCCESS")
            logging.info("\tConvert to ONNX Simplified: START")

    # Simplified ONNX convertation
    model_simp, check = simplify(onnx_model, skip_fuse_bn=True)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, export_simple_path)

    if verbose:
        logging.info("\tConvert to ONNX Simplified: SUCCESS")
        logging.info("\tResult validation: START")

    # Check the result

    try:
        ort_session = ort.InferenceSession(export_simple_path)
        outputs = ort_session.run(None, {"input": dummy_input.cpu().numpy()})
    except Exception as e:
        logging.info("\tCan\'t start onnxruntime session")
        raise e

    if outputs[0].shape != dummy_output.shape:
        if os.path.exists(export_path):
            os.remove(export_path)
        if os.path.exists(export_simple_path):
            os.remove(export_simple_path)
        raise Exception('Results validation: FAIL')

    if verbose:
        logging.info("\tResult validation: SUCCESS")
        logging.info(f"\t{sub_prefix}Result dim = {outputs[0].shape}")


def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    in_size = args.size
    model_type = args.model_type
    num_classes = args.num_classes
    model_path = args.model_path
    export_name = args.export_name
    verbose = args.verbose

    make_onnx_convertation(
        export_name=export_name,
        batch_size=batch_size,
        model_name=model_name,
        in_size=in_size,
        model_type=model_type,
        num_classes=num_classes,
        model_path=model_path,
        verbose=verbose,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX conversion script')
    parser.add_argument('--model-name', type=str, help="Name of the model", required=True)
    parser.add_argument('--model-type', default='classifier', type=str, help="Type of the model")
    parser.add_argument(
        '--model-path',
        type=str,
        default='ImageNet',
        help="Path to the pretrained weights, or ImageNet," "if you want to get model with imagenet pretrain",
    )
    parser.add_argument('--export-name', type=str, help="Name of the exported onnx file")
    parser.add_argument('--batch-size', type=int, default=1, help="Batch size for optimized model")
    parser.add_argument('--size', nargs='+', default=(3, 224, 224), type=int, help="Size of the input tensor")
    parser.add_argument('--num-classes', default=1000, type=int, help="Number of output classes of the model")
    parser.add_argument('--verbose', default=True, type=bool, help="Flag for showing information")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
