from timm.models.res2net import (
    res2net50_14w_8s,
    res2net50_26w_4s,
    res2net50_26w_6s,
    res2net50_26w_8s,
    res2net50_48w_2s,
    res2net101_26w_4s,
    res2next50,
)
from torch import nn

__all__ = [
    'timm_res2net50_26w_4s',
    'timm_res2net101_26w_4s',
    'timm_res2net50_26w_6s',
    'timm_res2net50_26w_8s',
    'timm_res2net50_48w_2s',
    'timm_res2net50_14w_8s',
    'timm_res2next50',
]


def timm_res2net50_26w_4s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_4s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net101_26w_4s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net101_26w_4s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_26w_6s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_6s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_26w_8s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_26w_8s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_48w_2s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_48w_2s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2net50_14w_8s(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2net50_14w_8s(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


def timm_res2next50(pretrained: bool, requires_grad: bool = True, **kwargs):
    model = res2next50(pretrained=pretrained, **kwargs)
    for params in model.parameters():
        params.requires_grad = requires_grad
    return model


class TestNetEncoder(nn.Module):
    def __init__(self, model, layers):
        super(TestNetEncoder, self).__init__()
        self.model = model
        self.layers = layers

        self.encoder_list = self._get_encoder()
        print(len(self.encoder_list))

    def _get_encoder(self):
        encoder_list = nn.ModuleList([])
        for i in range(5):
            encoder_layer = nn.ModuleList([])
            for layer in self.layers[i]:
                encoder_layer.append(dict(self.model.named_children())[layer])
            encoder_list.append(nn.Sequential(*encoder_layer))

        return encoder_list

    def forward(self, x):
        encoder_list = []
        for encoder_layer in self.encoder_list:
            x = encoder_layer(x)
            # print(x.shape)
            encoder_list.append(x)
        for i in encoder_list:
            print(i.shape)
        return x  # , encoder_list


if __name__ == '__main__':

    # _test()

    input_size = (3, 223, 224)
    # model = timm_res2net50_14w_8s(pretrained=False)
    # model = timm_res2net50_48w_2s(pretrained=False)
    # model = timm_res2net50_26w_4s(pretrained=False)
    # model = timm_res2net50_26w_6s(pretrained=False)
    # model = timm_res2net50_26w_8s(pretrained=False)
    # model = timm_res2next50(pretrained=False)
    model = timm_res2net101_26w_4s(pretrained=False)

    print(model)

    for i, (mk, mv) in enumerate(model.named_children()):
        print(i, mk)
        # if mk == 'features':
        #     for j, (fk, fv) in enumerate(mv.named_children()):
        #         print(i, j, fk)
        #
    TestNetEncoder_layers = (['conv1', 'bn1', 'act1'], ['maxpool', 'layer1'], ['layer2'], ['layer3'], ['layer4'])

    encoder = TestNetEncoder(model, layers=TestNetEncoder_layers)
    import torch
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    summary(encoder, input_size=input_size)
