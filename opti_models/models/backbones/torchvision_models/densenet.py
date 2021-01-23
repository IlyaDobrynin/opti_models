from torchvision import models


def densenet121(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.densenet121(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet161(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.densenet161(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet169(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.densenet169(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def densenet201(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.densenet201(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
