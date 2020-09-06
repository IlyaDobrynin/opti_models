# -*- coding: utf-8 -*-
"""
Simple backbones factory
"""
from . import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    mobilenetv2_w1, mobilenetv2_wd2, mobilenetv2_wd4, mobilenetv2_w3d4
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
