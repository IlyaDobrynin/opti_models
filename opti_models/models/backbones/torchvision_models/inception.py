from torchvision import models


def inception_v3(pretrained: str, progress: bool = True, requires_grad: bool = True):
    if pretrained == 'imagenet':
        pretrained = True
    else:
        pretrained = False
    model = models.inception_v3(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model