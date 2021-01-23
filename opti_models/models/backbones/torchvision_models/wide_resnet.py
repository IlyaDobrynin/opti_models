from torchvision import models


def wide_resnet50_2(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.wide_resnet50_2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def wide_resnet101_2(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.wide_resnet101_2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model