from torchvision import models


def vgg11(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg11(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg11_bn(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg11_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg13(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg13(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg13_bn(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg13_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg16(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg16(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg16_bn(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg16_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg19(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg19(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def vgg19_bn(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.vgg19_bn(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model