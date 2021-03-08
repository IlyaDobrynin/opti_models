from torchvision import models

__all__ = ['wide_resnet50_2', 'wide_resnet101_2']


def wide_resnet50_2(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.wide_resnet50_2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def wide_resnet101_2(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.wide_resnet101_2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model