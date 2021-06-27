from timm.models.resnet import ig_resnext101_32x48d, resnetrs420, swsl_resnext101_32x8d

__all__ = [
    'timm_ig_resnext101_32x48d',
    'timm_resnetrs420',
    'timm_swsl_resnext101_32x8d',
]


def timm_ig_resnext101_32x48d(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = ig_resnext101_32x48d(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_resnetrs420(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = resnetrs420(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_swsl_resnext101_32x8d(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = swsl_resnext101_32x8d(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
