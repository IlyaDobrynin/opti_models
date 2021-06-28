from timm.models.efficientnet import (
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b1_pruned,
    efficientnet_b2,
    efficientnet_b2_pruned,
    efficientnet_b2a,
    efficientnet_b3,
    efficientnet_b3_pruned,
    efficientnet_b3a,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    efficientnet_b8,
    efficientnet_cc_b0_4e,
    efficientnet_cc_b0_8e,
    efficientnet_cc_b1_8e,
    efficientnet_el,
    efficientnet_el_pruned,
    efficientnet_em,
    efficientnet_es,
    efficientnet_es_pruned,
    efficientnet_l2,
    efficientnet_lite0,
    efficientnet_lite1,
    efficientnet_lite2,
    efficientnet_lite3,
    efficientnet_lite4,
)

__all__ = [
    'timm_efficientnet_b0',
    'timm_efficientnet_b1',
    'timm_efficientnet_b2',
    'timm_efficientnet_b3',
    'timm_efficientnet_b4',
    'timm_efficientnet_b5',
    'timm_efficientnet_b6',
    'timm_efficientnet_b7',
    'timm_efficientnet_b8',
    'timm_efficientnet_b1_pruned',
    'timm_efficientnet_b2_pruned',
    'timm_efficientnet_b3_pruned',
    'timm_efficientnet_es',
    'timm_efficientnet_em',
    'timm_efficientnet_el',
    'timm_efficientnet_es_pruned',
    'timm_efficientnet_el_pruned',
    'timm_efficientnet_lite0',
    'timm_efficientnet_lite1',
    'timm_efficientnet_lite2',
    'timm_efficientnet_lite3',
    'timm_efficientnet_lite4',
    'timm_efficientnet_l2',
    'timm_efficientnet_b2a',
    'timm_efficientnet_b3a',
    'timm_efficientnet_cc_b0_4e',
    'timm_efficientnet_cc_b0_8e',
    'timm_efficientnet_cc_b1_8e',
]


def timm_efficientnet_b0(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b0(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b1(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b1(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b2(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b2(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b3(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b3(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b4(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b4(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b5(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b5(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b6(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b6(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b7(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b7(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b8(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b8(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b1_pruned(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b1_pruned(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b2_pruned(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b2_pruned(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b3_pruned(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b3_pruned(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_es(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_es(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_em(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_em(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_el(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_el(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_es_pruned(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_es_pruned(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_el_pruned(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_el_pruned(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_lite0(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_lite0(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_lite1(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_lite1(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_lite2(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_lite2(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_lite3(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_lite3(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_lite4(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_lite4(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_l2(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_l2(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b2a(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b2a(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_b3a(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_b3a(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_cc_b0_4e(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_cc_b0_4e(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_cc_b0_8e(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_cc_b0_8e(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_efficientnet_cc_b1_8e(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = efficientnet_cc_b1_8e(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model
