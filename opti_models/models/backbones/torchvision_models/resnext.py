from torchvision import models


def resnext50_32x4d(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.resnext50_32x4d(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnext101_32x8d(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.resnext101_32x8d(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model