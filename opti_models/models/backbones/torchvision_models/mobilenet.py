from torchvision import models


def mobilenet_v2(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.mobilenet_v2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
