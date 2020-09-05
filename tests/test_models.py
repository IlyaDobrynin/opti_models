from torchsummary import summary
from argparse import ArgumentParser
import logging
import torch
from opti_models.models import models_facade
logging.basicConfig(level=logging.INFO)


def test_classification(show: bool):
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

    passed_list = []
    failed_list = []
    for i, backbone_name in enumerate(backbone_names):
        try:
            model_parameters['backbone'] = backbone_name
            model = models_facade_obj.get_model_class(model_definition='basic_classifier')(**model_parameters)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            passed_list.append((backbone_name, 'passed'))
            if show:
                summary(model, input_size=input_size)
        except Exception as e:
            failed_list.append((backbone_name, e))

    if len(failed_list) == 0:
        logging.info("\tCLASSIFICATION TESTS PASSED")
    else:
        logging.info("\tCLASSIFICATION TESTS FAILED")

    for backbone_name, msg in failed_list + passed_list:
        logging.info(f"\t\tBACKBONE: {backbone_name}  MESSAGE: {msg}")


def parse_args():
    test_config_default = 'c'
    parser = ArgumentParser()
    parser.add_argument('--test_config', default=test_config_default, type=str, help="Type of tests to run")
    parser.add_argument('--show_models', default=False, type=bool, help="Flag to show models parameters")
    return parser.parse_args()


def main(args):
    if 'c' in args.test_config:
        test_classification(show=args.show_models)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)