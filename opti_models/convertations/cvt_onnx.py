import os
import typing as t
import onnx
import torch
from torch.nn import Module
import onnxruntime as ort
from onnxsim import simplify
import argparse
import logging
from torchsummary import summary
from opti_models.models import models_facade

logging.basicConfig(level=logging.INFO)
sub_prefix = ">>>>> "

__all__ = ["make_onnx_convertation"]


def _patch_last_linear(model: Module, num_classes: int):
    layers = list(model.named_children())
    layer_full_name = []
    try:
        last_layer_name, last_layer = layers[-1]
        layer_full_name.append(last_layer_name)
        while not isinstance(last_layer, torch.nn.Linear):
            last_layer_name, last_layer = list(last_layer.named_children())[-1]
            layer_full_name.append(last_layer_name)
    except TypeError:
        raise TypeError("Can't find linear layer in the model")

    features_dim = last_layer.in_features
    res_attr = model
    for layer_attr in layer_full_name[:-1]:
        res_attr = getattr(res_attr, layer_attr)
    setattr(res_attr, layer_full_name[-1], torch.nn.Linear(features_dim, num_classes))


def get_model(
        model_type: str,
        model_name: str,
        num_classes: t.Optional[int] = 1,
        model_path: t.Optional[str] = None,
        classifier_params: t.Optional[t.Dict] = None,
        show: bool = False,
        input_shape: t.Optional[t.Tuple] = (224, 224, 3),
) -> torch.nn.Module:
    if model_type == 'default':
        if model_path == 'ImageNet':
            pretrained = True
        else:
            pretrained = False
        m_facade = models_facade.ModelFacade(task='backbones')
        parameters = dict(requires_grad=True, pretrained=pretrained)
        model = m_facade.get_model_class(model_definition=model_name)(**parameters)

        # Patch last linear layer if needed
        if num_classes != 1000:
            _patch_last_linear(model=model, num_classes=num_classes)

    elif model_type == 'classifier':
        m_facade = models_facade.ModelFacade(task='classification')
        if model_path == 'ImageNet':
            pretrained = True
        else:
            pretrained = False
        model_params = classifier_params if classifier_params \
            else dict(backbone=model_name, depth=5, num_classes=num_classes, pretrained=pretrained)
        model = m_facade.get_model_class(model_definition='basic_classifier')(**model_params)
    # TODO: Add segmentation, detection, OCR tasks
    else:
        raise NotImplementedError(
            f"Model type {model_type} not implemented."
            f"Use one of ['default', 'default_patch', 'classifier']"
        )

    if model_path:
        model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    if show:
        summary(model, input_size=input_shape)
    return model


def get_parameters(export_dir: str, model_name: str, batch_size: int, in_size: t.Tuple) -> t.Tuple:
    export_dir = os.path.join(export_dir, model_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    out_model_name = f"{model_name}_bs-{batch_size}_res-{in_size[0]}x{in_size[1]}"
    export_path = os.path.join(export_dir, f"{out_model_name}.onnx")
    export_simple_path = os.path.join(export_dir, f"{out_model_name}_simplified.onnx")
    input_size = (batch_size, 3, in_size[0], in_size[1])
    return export_path, export_simple_path, input_size


def make_onnx_convertation(
        export_dir: str,
        batch_size: int,
        model_name: str,
        in_size: t.Tuple,
        model_type: str,
        num_classes: int,
        model_path: t.Optional[str] = None,
        model: t.Optional[torch.nn.Module] = None,
        verbose: t.Optional[bool] = False
):
    if verbose:
        logging.info("\tConvert to ONNX: START")

    export_path, export_simple_path, input_size = get_parameters(
        export_dir=export_dir,
        batch_size=batch_size,
        model_name=model_name,
        in_size=in_size
    )
    if model_type != 'custom':
        model = get_model(
            model_name=model_name,
            input_shape=input_size[1:],
            model_type=model_type,
            num_classes=num_classes,
            model_path=model_path
        )
    else:
        if model is None:
            raise NotImplemented(
                'Parameter model_mode is set to "custom", but model not specified.'
            )
        model.load_state_dict(torch.load(model_path))

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
    except:
        print('Convert to ONNX: FAIL')
    else:
        if verbose:
         logging.info("\tConvert to ONNX: SUCCESS")

    try:
        onnx_model = onnx.load(export_path)
        onnx.checker.check_model(onnx_model)
    except:
        print('ONNX check: FAIL')
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
    except:
        print('Can\'t start onnxruntime session')

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
    model_type = args.model_mode
    num_classes = args.num_classes
    model_path = args.model_path
    export_dir = args.export_dir
    verbose = args.verbose

    make_onnx_convertation(
        export_dir=export_dir,
        batch_size=batch_size,
        model_name=model_name,
        in_size=in_size,
        model_type=model_type,
        num_classes=num_classes,
        model_path=model_path,
        verbose=verbose
    )


def parse_args():
    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('model-name', type=str, help="Name of the model")
    parser.add_argument('model-mode', default='default', type=str, help="Mode of the model")
    parser.add_argument('--model-path', type=str, default=None, help="Path to the pretrained weights, or ImageNet,"
                                                                     "if you want to get model with imagenet pretrain")
    parser.add_argument('--export-dir', default='data/onnx-export', help="Path to directory to save results")
    parser.add_argument('--batch-size', default=1, help="Batch size for optimized model")
    parser.add_argument('--size', nargs='+', default=(224, 224, 3), type=int, help="Size of the input tensor")
    parser.add_argument('--num-classes', default=1000, type=int, help="Number of output classes of the model")
    parser.add_argument('--verbose', default=True, type=bool, help="Flag for showing information")

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

    for name in model_names:
        logging.info(f"{name.upper()} CONVERT")
        args = parse_args()
        args.model_name = name
        main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
