import torchvision.models as models


__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def resnet18(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.resnet18(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet34(pretrained: bool = True, progress: bool = True, requires_grad: bool = True):
    model = models.resnet34(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet50(pretrained: bool = True, progress: bool = True, requires_grad: bool = True):
    model = models.resnet50(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet101(pretrained: bool = True, progress: bool = True, requires_grad: bool = True):
    model = models.resnet101(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def resnet152(pretrained: bool = True, progress: bool = True, requires_grad: bool = True):
    model = models.resnet152(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
