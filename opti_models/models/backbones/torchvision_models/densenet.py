from torchvision import models


def densenet121(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.densenet121(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet161(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.densenet161(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet169(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.densenet169(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet201(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.densenet201(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
