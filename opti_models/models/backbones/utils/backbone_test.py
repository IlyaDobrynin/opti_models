from torch import nn


class TestNetEncoder(nn.Module):
    def __init__(self, model, layers):
        super(TestNetEncoder, self).__init__()
        self.model = model
        self.layers = layers

        self.encoder_list = self._get_encoder()
        print(len(self.encoder_list))

    def _get_encoder(self):
        encoder_list = nn.ModuleList([])

        if 'features' in dict(self.model.named_children()).keys():
            for (mk, mv) in self.model.named_children():
                if mk == 'features':
                    for i in range(len(self.layers)):
                        encoder_layer = nn.ModuleList([])
                        for layer in self.layers[i]:
                            print(layer)
                            encoder_layer.append(dict(mv.named_children())[layer])
                        encoder_list.append(nn.Sequential(*encoder_layer))
                else:
                    continue
        else:
            encoder_list = nn.ModuleList([])
            for i in range(len(self.layers)):
                encoder_layer = nn.ModuleList([])
                for layer in self.layers[i]:
                    encoder_layer.append(dict(self.model.named_children())[layer])
                encoder_list.append(nn.Sequential(*encoder_layer))

        del self.model
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
    import torch
    from torchsummary import summary

    from opti_models.models.backbones.timm_models.timm_resnet import (
        timm_swsl_resnext101_32x8d,
    )

    input_size = [3, 224, 224]
    model = timm_swsl_resnext101_32x8d(pretrained=False)
    print(model)

    for i, (mk, mv) in enumerate(model.named_children()):
        print(i, mk)
        if mk in ['features', 'stages']:
            for j, (fk, fv) in enumerate(mv.named_children()):
                print(i, j, fk)

    from opti_models.models.backbones.utils.skip_names import RESNEST_LAYERS

    TestNetEncoder_layers = RESNEST_LAYERS

    encoder = TestNetEncoder(model, layers=TestNetEncoder_layers)
    encoder.forward(torch.zeros(size=[1] + input_size))

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = encoder.to(device)
    # summary(encoder, input_size=input_size)
