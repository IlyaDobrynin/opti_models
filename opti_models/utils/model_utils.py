import logging
import os
import typing as t

import torch
from torch.nn import Module
from torchsummary import summary

from ..models import models_facade

logging.basicConfig(level=logging.INFO)

__all__ = ["get_model"]


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
    model_name: str = None,
    num_classes: t.Optional[int] = 1000,
    input_shape: t.Optional[t.Tuple] = (3, 224, 224),
    model: t.Optional[Module] = None,
    model_path: t.Optional[str] = None,
    classifier_params: t.Optional[t.Dict] = None,
    show: bool = False,
) -> torch.nn.Module:
    """Function returns model class

    Args:
        model_type: One of the model types. Available: 'classifier', 'opti-classifier', 'custom'. If 'custom',
                    tou shall provide model class as 'model' argument.
        model_name: One of the predefined backbones name (see opti_models.show_available_backbones() for details).
        num_classes: Number of model output classes. Default = 1000. This parameter ignored,
                    if provide custom model class
        input_shape: Tuple with imput shape of the model. Default = (3, 224, 224)
        model: Custom model class. If provided, returns as is.
        model_path: Path to the custom model weights.
        classifier_params: Parameters for classifier, if model_type == 'opti-classifier'
        show:

    Returns:
        model: Model class, located on the cuda device (if available).
    """
    if model_type == 'classifier':
        if isinstance(model_path, str) and model_path.lower() == 'imagenet':
            pretrained = True
        else:
            pretrained = False
        m_facade = models_facade.ModelFacade(task='classification')
        parameters = dict(requires_grad=True, pretrained=pretrained)
        model = m_facade.get_model_class(model_definition=model_name)(**parameters)

        # Patch last linear layer if needed
        if num_classes is not None and num_classes != 1000:
            _patch_last_linear(model=model, num_classes=num_classes)

    elif model_type == 'opti-classifier':
        m_facade = models_facade.ModelFacade(task='opti-classification')
        if model_path.lower() == 'imagenet':
            pretrained = model_path.lower()
        else:
            pretrained = None

        if classifier_params is not None:
            model_params = classifier_params
        else:
            model_params = dict(
                backbone=model_name,
                depth=5,
                num_classes=num_classes,
                num_input_channels=input_shape[0],
                num_last_filters=128,
                dropout=0.2,
                pretrained=pretrained,
                unfreeze_encoder=True,
                custom_enc_start=False,
                use_complex_final=False,
                conv_type='default',
                bn_type='default',
                activation_type='relu',
                depthwise=False,
            )
            logging.info(f"\tArgument classifier_params is empty, use default:\n\t{model_params}")
        model = m_facade.get_model_class(model_definition='basic_classifier')(**model_params)
    elif model_type == 'custom':
        if model is None:
            raise NotImplementedError('Parameter model_mode is set to "custom", but model not specified.')
    # TODO: Add segmentation, detection, OCR tasks
    else:
        raise NotImplementedError(
            f"Model type {model_type} not implemented." f"Use one of ['classifier', 'opti-classifier', 'custom']"
        )

    if isinstance(model_path, str) and model_path != 'ImageNet':
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path))
            except RuntimeError:
                model.load_state_dict(torch.load(model_path)['model_state_dict'])
            except Exception:
                raise RuntimeError(
                    'Please provide model weights either as the whole file, '
                    'or as a \'model_state_dict\' part of the file'
                )
        else:
            raise FileNotFoundError(f"No such file or directory: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if show:
        summary(model, input_size=input_shape)
    return model
