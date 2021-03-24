from torchvision import models

__all__ = ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']


def mnasnet0_5(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.mnasnet0_5(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def mnasnet0_75(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.mnasnet0_75(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def mnasnet1_0(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.mnasnet1_0(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def mnasnet1_3(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.mnasnet1_3(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
