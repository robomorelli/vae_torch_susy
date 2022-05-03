from torch import nn
import torch
import torch.nn.functional as F
import math

class InverseSquareRootLinearUnit(nn.Module):

    def __init__(self, min_value=5e-3):
        super(InverseSquareRootLinearUnit, self).__init__()
        self.min_value = min_value

    def forward(self, x):
        return 1. + self.min_value \
               + torch.where(torch.gt(x, 0), x, torch.div(x, torch.sqrt(1 + (x * x))))

class ClippedTanh(nn.Module):

    def __init__(self, min_value=5e-3):
        super(ClippedTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (1 + 0.999 * torch.tanh(x))

class SmashTo0(nn.Module):

    def __init__(self):
        super(SmashTo0, self).__init__()

    def forward(self, x):
        return 0*x

class h1_prior(nn.Module):
    def __init__(self, in_features, out_features):
        super(h1_prior, self).__init__()
        self.h1_prior_w = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.h1_prior_b = nn.Parameter(torch.ones(out_features), requires_grad=False)

    def forward(self, x):
        x = F.linear(x, self.h1_prior_w, self.h1_prior_b)
        return x

#The usual way to do this is to use the functional interface to redefine the forward

class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return F.conv2d(input, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ConstrainedDec(nn.Linear):
    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class Dec1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dec1, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _max_norm(self, w):
        norm = w.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, self._max_norm_val)
        return w * (desired / (self._eps + norm))

    def forward(self, x):
        x = F.linear(x, self.weight.clamp(min=-1.0, max=1.0), self.bias)
        return x

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        return x

'''
class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
'''