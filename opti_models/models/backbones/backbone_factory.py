# -*- coding: utf-8 -*-
"""
Simple backbones factory
"""
from . import (
    # Torchvision models
    resnet18, resnet34, resnet50, resnet101, resnet152,
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    densenet121, densenet161, densenet169, densenet201,
    inception_v3,
    resnext50_32x4d, resnext101_32x8d,
    wide_resnet50_2, wide_resnet101_2,
    mobilenet_v2,

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
    # Torchvision models
    "resnet152": resnet152,
    "resnet101": resnet101,
    "resnet50": resnet50,
    "resnet34": resnet34,
    "resnet18": resnet18,
    'mobilenet_v2': mobilenet_v2,
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet161': densenet161,
    'densenet201': densenet201,
    'inception_v3': inception_v3,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50': wide_resnet50_2,
    'wide_resnet101': wide_resnet101_2,

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


def show_available_backbones():
    return list(BACKBONES.keys())
