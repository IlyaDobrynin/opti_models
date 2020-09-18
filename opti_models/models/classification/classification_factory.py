# -*- coding: utf-8 -*-
"""
Module implements classification autobuilder class

"""
import gc
from torch import nn
from collections import OrderedDict
from ..backbones import backbone_factory
from ..backbones.utils.backbones_meta import encoder_dict
from ..custom_blocks.common_blocks import ConvBnRelu
from ..utils import patch_first_conv


class ClassificationFactory(nn.Module):
    def __init__(
            self,
            backbone: str,
            depth: int,
            num_classes: int,
            num_input_channels: int = 3,
            num_last_filters: int = 128,
            dropout: float = 0.5,
            pretrained: str = 'imagenet',
            unfreeze_encoder: bool = True,
            custom_enc_start: bool = False,
            use_complex_final: bool = True,
            conv_type: str = 'default',
            bn_type: str = 'default',
            activation_type: str = 'relu',
            depthwise: bool = False
    ):

        super().__init__()

        assert backbone in backbone_factory.BACKBONES.keys(), \
            f"Wrong name of backbone: {backbone}. " \
                f"Should be in {backbone_factory.BACKBONES.keys()}."

        self.num_input_channels = num_input_channels
        self.depthwise = depthwise
        self.bn_type = bn_type
        self.conv_type = conv_type
        self.activation_type = activation_type
        self.depth = depth
        self.backbone = backbone

        self.encoder = backbone_factory.get_backbone(
            name=self.backbone,
            pretrained=pretrained,
            requires_grad=unfreeze_encoder
        )
        if num_input_channels != 3:
            patch_first_conv(model=self.encoder, in_channels=num_input_channels)
        self.encoder_layers_dict = encoder_dict[backbone]['skip']
        self.encoder_filters = encoder_dict[backbone]['filters']
        self.is_featured = encoder_dict[backbone]['features']

        if custom_enc_start:
            first_enc_layer = nn.Sequential(
                OrderedDict(
                    [
                        ("first_enc_conv_bn_relu", ConvBnRelu(
                            in_channels=self.num_input_channels,
                            out_channels=self.encoder_filters[0],
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            depthwise=self.depthwise,
                            bn_type=self.bn_type,
                            conv_type=self.conv_type,
                            activation_type=activation_type
                        ))
                    ]
                )
            )
        else:
            first_enc_layer = None
        self.encoder_layers = self._get_encoder(first_enc_layer)
        self.classifier_dropout = nn.Dropout2d(p=dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        if use_complex_final:
            self.last_layers = nn.Sequential(
                nn.Linear(self.encoder_filters[depth - 1], num_last_filters),
                nn.ReLU(inplace=True),
                nn.Linear(num_last_filters, num_classes)
            )
        else:
            self.last_layers = nn.Linear(self.encoder_filters[depth - 1], num_classes)

    def _get_encoder(self, first_enc_layer):
        """ Function to define encoder layers

        :param: first_enc_layer
        :return: List of encoder layers
        """
        encoder_list = nn.ModuleList([])
        if self.is_featured:
            for (mk, mv) in self.encoder.named_children():
                if mk in ['features', 'encoder_layers', 'module_list']:
                    if first_enc_layer is None:
                        for i in range(self.depth):
                            encoder_layer = nn.ModuleList([])
                            for layer in self.encoder_layers_dict[i]:
                                encoder_layer.append(dict(mv.named_children())[layer])
                            encoder_list.append(nn.Sequential(*encoder_layer))
                    else:
                        encoder_list.append(first_enc_layer)
                        for i in range(1, self.depth):
                            encoder_layer = nn.ModuleList([])
                            for layer in self.encoder_layers_dict[i]:
                                encoder_layer.append(dict(mv.named_children())[layer])
                            encoder_list.append(nn.Sequential(*encoder_layer))
                else:
                    continue
        else:
            if first_enc_layer is None:
                for i in range(self.depth):
                    encoder_layer = nn.ModuleList([])
                    for layer in self.encoder_layers_dict[i]:
                        encoder_layer.append(dict(self.encoder.named_children())[layer])
                    encoder_list.append(nn.Sequential(*encoder_layer))
            else:
                encoder_list.append(first_enc_layer)
                for i in range(1, self.depth):
                    encoder_layer = nn.ModuleList([])
                    for layer in self.encoder_layers_dict[i]:
                        encoder_layer.append(dict(self.encoder.named_children())[layer])
                    encoder_list.append(nn.Sequential(*encoder_layer))
        del self.encoder
        gc.collect()
        return encoder_list

    def _make_encoder_forward(self, x):
        """ Function to make u-net encoder

        :param x: Input tenzor
        :return: List of encoder tensors
        """
        encoder_list = []
        if self.backbone in ['pnasnet5large', 'nasnetalarge']:
            encoder_list_tmp = []
            counter = 2
            # for i, outer_layer in enumerate(self.features):
            for i, outer_layer in enumerate(self.encoder_layers):
                if i < 2:
                    x = outer_layer(x)
                    encoder_list.append(x.clone())
                    encoder_list_tmp.append(x.clone())
                    continue
                else:
                    for inner_layer in outer_layer:
                        if self.backbone == 'nasnetalarge':
                            first_layer = encoder_list_tmp[counter - 1]
                            if counter == 2:
                                first_layer = encoder_list_tmp[counter - 2]
                                second_layer = encoder_list_tmp[counter - 1]
                            elif counter in (10, 17):
                                second_layer = encoder_list_tmp[counter - 3]
                            else:
                                second_layer = encoder_list_tmp[counter - 2]
                        else:
                            first_layer = encoder_list_tmp[counter - 2]
                            second_layer = encoder_list_tmp[counter - 1]
                        x = inner_layer(first_layer, second_layer)
                        encoder_list_tmp.append(x.clone())
                        counter += 1
                    encoder_list.append(x.clone())
            del encoder_list_tmp
            gc.collect()
        else:
            # for encoder_layer in self.features:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encoder_list.append(x)
        return encoder_list

    def forward(self, x):
        encoder_list = self._make_encoder_forward(x)
        out = encoder_list[-1]
        classifier_out = self.avgpool(out)
        _, c, h, w = classifier_out.data.size()
        classifier_out = classifier_out.view(-1, c * h * w)
        classifier_out = self.classifier_dropout(classifier_out)
        classifier_out = self.last_layers(classifier_out)

        return classifier_out

