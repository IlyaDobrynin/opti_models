# -*- coding: utf-8 -*-
"""
Simple backbones factory
"""
from . import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenetv2_w1, mobilenetv2_wd2, mobilenetv2_wd4, mobilenetv2_w3d4,
    mobilenetv3_large_w1,
    mixnet_s, mixnet_m, mixnet_l,
    efficientnet_b0, efficientnet_b1,
    efficientnet_b0b, efficientnet_b1b, efficientnet_b2b, efficientnet_b3b, efficientnet_b4b,
    efficientnet_b5b, efficientnet_b6b, efficientnet_b7b,
    efficientnet_b0c, efficientnet_b1c, efficientnet_b2c, efficientnet_b3c, efficientnet_b4c,
    efficientnet_b5c, efficientnet_b6c, efficientnet_b7c, efficientnet_b8c,
    genet_small, genet_normal, genet_large
)


BACKBONES = {
    # Resnets
    "resnet152": resnet152,
    "resnet101": resnet101,
    "resnet50": resnet50,
    "resnet34": resnet34,
    "resnet18": resnet18,

    # Mobilenets
    "mobilenetv2_w1": mobilenetv2_w1,
    "mobilenetv2_wd2": mobilenetv2_wd2,
    "mobilenetv2_wd4": mobilenetv2_wd4,
    "mobilenetv2_w3d4": mobilenetv2_w3d4,
    "mobilenetv3_large_w1": mobilenetv3_large_w1,

    # MixNet
    "mixnet_s": mixnet_s,
    "mixnet_m": mixnet_m,
    "mixnet_l": mixnet_l,

    # Efficientnets
    'efficientnet_b0': efficientnet_b0,
    'efficientnet_b1': efficientnet_b1,

    'efficientnet_b0b': efficientnet_b0b,
    'efficientnet_b1b': efficientnet_b1b,
    'efficientnet_b2b': efficientnet_b2b,
    'efficientnet_b3b': efficientnet_b3b,
    'efficientnet_b4b': efficientnet_b4b,
    'efficientnet_b5b': efficientnet_b5b,
    'efficientnet_b6b': efficientnet_b6b,
    'efficientnet_b7b': efficientnet_b7b,

    'efficientnet_b0c': efficientnet_b0c,
    'efficientnet_b1c': efficientnet_b1c,
    'efficientnet_b2c': efficientnet_b2c,
    'efficientnet_b3c': efficientnet_b3c,
    'efficientnet_b4c': efficientnet_b4c,
    'efficientnet_b5c': efficientnet_b5c,
    'efficientnet_b6c': efficientnet_b6c,
    'efficientnet_b7c': efficientnet_b7c,
    'efficientnet_b8c': efficientnet_b8c,

    'genet_small': genet_small,
    'genet_normal': genet_normal,
    'genet_large': genet_large,

}


def get_backbone(name, *args, **kwargs):
    """ Function returns pytorch pretrained model with given args and kwargs
    from the list of backbones

    :param name: Pretrained backbone name
    :param args: Model arguments
    :param kwargs: Model keyword arguments
    :return: Model
    """
    return BACKBONES[name](*args, **kwargs)
