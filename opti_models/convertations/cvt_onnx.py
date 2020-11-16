import os
import typing as t
import onnx
import torch
import torchvision
import onnxruntime as ort
from onnxsim import simplify
import argparse
import logging
from torchsummary import summary
from opti_models.models import models_facade
from opti_models.models import t_models

logging.basicConfig(level=logging.INFO)
sub_prefix = ">>>>> "

__all__ = ["make_onnx_convertation"]



def get_model(model_name: str, input_shape: t.Tuple, model_mode: str, num_classes: int, ckpt_path: str, show: bool = False) -> torch.nn.Module:
    m_facade = models_facade.ModelFacade(task='classification')
    if model_mode == 'torchvision':
        if ckpt_path == 'imagenet':
            assert model_name in t_models.T_BACKBONES.keys(), 'Specify correct model_name. For {} should be one of the following: {}'.format(model_mode, t_models.show_available_backbones()[0])
            model = t_models.T_BACKBONES[model_name](pretrained=True)
            
        else:
            model = torchvision.models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(torch.load(ckpt_path))
    else:
        if ckpt_path == 'imagenet':
            pretrained = ckpt_path
        else:
            pretrained = None

        model_params = dict(
                backbone=model_name,
                depth=5,
                num_classes=num_classes,
                pretrained=pretrained)
        model = m_facade.get_model_class(
            model_definition='basic_classifier')(**model_params)
        
        if ckpt_path != 'imagenet':
            model.load_state_dict(torch.load(ckpt_path))

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
        model_mode:str,
        num_classes: int,
        ckpt_path: str,
        info: bool = False
):
    if info:
        logging.info("\tConvert to ONNX: START")

    export_path, export_simple_path, input_size = get_parameters(
        export_dir=export_dir,
        batch_size=batch_size,
        model_name=model_name,
        in_size=in_size
    )
    if model_mode in ['torchvision', 'opti']:
        model = get_model(
                model_name=model_name,
                input_shape=input_size[1:],
                model_mode=model_mode,
                num_classes=num_classes,
                ckpt_path=ckpt_path
            )
    elif model_mode == 'custom':
        # Implement your own Model Class
        model = None
        
        if model == None:
            raise NotImplemented('Implement your own Model Class') 
        model.load_state_dict(torch.load(ckpt_path))
    else:
        raise Exception('Specify correct model_mode. Should be one of the following: [\'opti\', \'torchvision\', \'custom\']')
    

    dummy_input = torch.ones(input_size, device="cuda")
    with torch.no_grad():
        pre_det_res = model(dummy_input.cuda())
    dummy_output = torch.ones(*pre_det_res.shape, device="cuda")

    input_names = ["input"]
    output_names = ["output"]
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
    if info:
        logging.info("\tConvert to ONNX: SUCCESS")

    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)

    if info:
        logging.info("\tONNX check: SUCCESS")
        logging.info("\tConvert to ONNX Simplified: START")

    # Simplified ONNX convertation
    model_simp, check = simplify(onnx_model, skip_fuse_bn=True)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, export_simple_path)

    if info:
        logging.info("\tConvert to ONNX Simplified: SUCCESS")
        logging.info("\tResult validation: START")

    # Check the result
    ort_session = ort.InferenceSession(export_simple_path)
    outputs = ort_session.run(None, {"input": dummy_input.cpu().numpy()})

    if outputs[0].shape != dummy_output.shape:
        if os.path.exists(export_path):
            os.remove(export_path)
        if os.path.exists(export_simple_path):
            os.remove(export_simple_path)
        raise Exception('Results validation: FAILED')

    if info:
        logging.info("\tResult validation: SUCCESS")
        logging.info(f"\t{sub_prefix}Result dim = {outputs[0].shape}")


def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    in_size = args.in_size
    model_mode = args.model_mode
    num_classes = args.num_classes
    ckpt_path = args.ckpt_path
    export_dir = args.export_dir
    info = args.info

    make_onnx_convertation(
        export_dir=export_dir,
        batch_size=batch_size,
        model_name=model_name,
        in_size=in_size,
        model_mode=model_mode,
        num_classes=num_classes,
        ckpt_path=ckpt_path,
        info=info
    )


def parse_args():
    model_name = "mobilenetv3_large_w1"
    export_dir = f"../../data/onnx_export"
    in_size = (224, 224)
    parser = argparse.ArgumentParser(description='TRT params')
    parser.add_argument('--model_name', default=model_name, type=str)
    parser.add_argument('--export_dir', default=export_dir)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--in_size', nargs="+", default=in_size, type=int)
    parser.add_argument('--model_mode', default='torchvision', type=str)
    parser.add_argument('--num_classes', default=1000, type=int)
    parser.add_argument('--ckpt_path', default='imagenet', type=str)
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

    for name in model_names:
        logging.info(f"{name.upper()} CONVERT")
        args = parse_args()
        args.model_name = name
        main(args=args)
        logging.info(f"-" * 100)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
