from timm.models.resnest import resnest14d

__all__ = [
    'timm_resnest14d',
]


def timm_resnest14d(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = resnest14d(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
