from timm.models.res2net import (
    res2net50_14w_8s,
    res2net50_26w_4s,
    res2net50_26w_6s,
    res2net50_26w_8s,
    res2net50_48w_2s,
    res2net101_26w_4s,
    res2next50,
)

from opti_models.models.backbones.utils.backbone_test import TestNetEncoder

__all__ = [
    'timm_res2net50_26w_4s',
    'timm_res2net101_26w_4s',
    'timm_res2net50_26w_6s',
    'timm_res2net50_26w_8s',
    'timm_res2net50_48w_2s',
    'timm_res2net50_14w_8s',
    'timm_res2next50',
]


def timm_res2net50_26w_4s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_4s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net101_26w_4s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net101_26w_4s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_26w_6s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_6s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_26w_8s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_8s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_48w_2s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_48w_2s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_14w_8s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_14w_8s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2next50(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2next50(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
