# -*- coding: utf-8 -*-
"""
Simple backbones factory
"""
from . import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)


BACKBONES = {
    'resnet152': resnet152,
    'resnet101': resnet101,
    'resnet50': resnet50,
    'resnet34': resnet34,
    'resnet18': resnet18,
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
