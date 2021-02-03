from torchvision import models

__all__ = ['mobilenet_v2']


def mobilenet_v2(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.mobilenet_v2(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
