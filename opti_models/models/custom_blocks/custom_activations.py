from torch import nn
import torch
import torch.nn.functional as F


class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        super().__init__()

    def mish(self, input):
        return input * torch.tanh(F.softplus(input))

    def forward(self, input):
        return self.mish(input)
