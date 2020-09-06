from torch import nn
from torch.nn import functional as F
from .custom_convolutions import DepthwiseConv2d, PartialConv2d, Conv2dSamePad
from .custom_activations import Mish


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Conv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            depthwise: bool = False,
            make_init: bool = True,
            conv_type: str = 'default'
    ):
        super(Conv, self).__init__()
        conv_parameters = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        if depthwise:
            self.conv = DepthwiseConv2d(conv_type=conv_type, make_init=make_init, **conv_parameters)
        else:
            if conv_type == 'default':
                self.conv = nn.Conv2d(**conv_parameters)
            elif conv_type == 'partial':
                self.conv = PartialConv2d(**conv_parameters)
            elif conv_type == 'same':
                self.conv = Conv2dSamePad(**conv_parameters)
            else:
                raise ValueError(
                    f'Wrong type of convolution: {conv_type}. '
                    f'Should be "default" or "partial"'
                )
        if make_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) \
                    or isinstance(m, PartialConv2d) \
                    or isinstance(m, Conv2dSamePad):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        return x


class Activation(nn.Module):
    def __init__(self, activation_type: str = 'relu'):
        super(Activation, self).__init__()
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'mish':
            self.activation = Mish()
        else:
            raise ValueError(
                f"Activation {activation_type} not implement"
            )

    def forward(self, x):
        x = self.activation(x)
        return x


class BatchNorm(nn.Module):
    def __init__(
            self,
            channels: int,
            bn_type: str,
            make_init: bool = True,
            **kwargs
    ):
        super(BatchNorm, self).__init__()
        if bn_type == 'default':
            self.bn = nn.BatchNorm2d(num_features=channels)
        elif bn_type == 'group':
            self.bn = nn.GroupNorm(
                num_groups=kwargs['num_groups'],
                num_channels=channels
            )
        else:
            raise ValueError(
                f'Wrong type if bn: {bn_type}. '
                f'Should be "default" or "sync"'
            )
        if make_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            add_relu: bool = True,
            interpolate: bool = False,
            depthwise: bool = False,
            conv_type: str = 'default',
            bn_type: str = 'default',
            activation_type: str = 'relu',
            make_init: bool = True
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
            depthwise=depthwise,
            conv_type=conv_type,
            make_init=make_init
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = BatchNorm(channels=out_channels, bn_type=bn_type, make_init=make_init)

        self.activation = Activation(
            activation_type=activation_type
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x