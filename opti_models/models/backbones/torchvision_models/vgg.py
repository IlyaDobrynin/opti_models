from torchvision import models


def vgg11(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg11(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg11_bn(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg11_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg13(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg13(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg13_bn(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg13_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg16(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg16(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg16_bn(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg16_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg19(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg19(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg19_bn(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.vgg19_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model