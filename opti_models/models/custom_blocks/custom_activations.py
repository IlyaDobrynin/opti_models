import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def mish(self, input):
        return input * torch.tanh(F.softplus(input))

    def forward(self, input):
        return self.mish(input)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0
