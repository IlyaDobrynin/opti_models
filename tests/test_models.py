from torchsummary import summary
from argparse import ArgumentParser
import torch
from opti_models.models import models_facade


def test_classification():
    backbone_names = [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    ]
    input_size = (3, 224, 224)

    model_parameters = dict(
        depth=5, num_classes=1000, num_input_channels=input_size[0], num_last_filters=256,
        dropout=0.5, pretrained='imagenet', unfreeze_encoder=False, custom_enc_start=False,
        use_complex_final=True, conv_type='default', bn_type='default', activation_type='relu',
        depthwise=False
    )
    models_facade_obj = models_facade.ModelFacade(task='classification')
    for backbone_name in backbone_names:
        model_parameters['backbone'] = backbone_name
        model = models_facade_obj.get_model_class(model_definition='basic_classifier')(**model_parameters)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        summary(model, input_size=input_size)


def parse_args():
    test_config_default = 'c'
    parser = ArgumentParser()
    parser.add_argument('--test_config', default=test_config_default, type=str, help="Type of tests to run")
    return parser.parse_args()


def main(args):
    test_classification()


if __name__ == '__main__':
    args = parse_args()
    main(args=args)