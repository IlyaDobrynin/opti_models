from torchvision import models


def inception_v3(pretrained: bool, progress: bool = True, requires_grad: bool = True):
    model = models.inception_v3(pretrained=pretrained, progress=progress)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model