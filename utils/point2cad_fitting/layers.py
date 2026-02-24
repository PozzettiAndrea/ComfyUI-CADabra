"""
Neural network layers for SplineINR.
Adapted from Point2CAD: https://github.com/prs-eth/point2cad
"""

import numpy as np
import torch
import warnings


class SinAct(torch.nn.Module):
    def forward(self, x):
        return x.sin()


class SincAct(torch.nn.Module):
    def forward(self, x):
        return x.sinc()


class CustomLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        bound_weight = kwargs.pop("bound_weight", None)
        bound_bias = kwargs.pop("bound_bias", None)
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            if bound_weight is not None:
                self.weight.uniform_(-bound_weight, bound_weight)
            if bound_bias is not None:
                self.weight.uniform_(-bound_bias, bound_bias)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_freqs, concat_input=True, dtype=torch.float32):
        super().__init__()
        if type(num_freqs) is not int or num_freqs < 0:
            raise ValueError(f"Invalid number of frequencies: {num_freqs}")
        if num_freqs == 0 and not concat_input:
            raise ValueError("Invalid combination of layer parameters")
        self.num_freqs = num_freqs
        self.concat_input = concat_input
        self.dtype = dtype
        if num_freqs > 0:
            self.register_buffer("freq_bands", 2 ** torch.arange(num_freqs, dtype=dtype))

    @property
    def dim_multiplier(self):
        return self.num_freqs * 2 + (1 if self.concat_input else 0)

    def forward(self, x):
        if not torch.is_tensor(x) or x.dim() < 2 or x.dtype != self.dtype:
            raise ValueError("Invalid input")
        B, D = x.shape[:-1], x.shape[-1]
        out = []
        if self.concat_input:
            out = [x]
        if self.num_freqs > 0:
            x = x.unsqueeze(-1)
            x = x * self.freq_bands
            x = x.reshape(B + (self.num_freqs * D,))
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            out.append(x)
        out = torch.cat(out, dim=-1)
        return out


class SirenLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, is_first=False, omega=30, act_type="sinc"):
        super().__init__()
        self.omega = omega
        if is_first:
            bound_weight = 1 / dim_in
        else:
            bound_weight = np.sqrt(6 / dim_in) / self.omega
        bound_bias = 0.1 * bound_weight
        self.linear = CustomLinear(dim_in, dim_out, bound_weight=bound_weight, bound_bias=bound_bias)
        self.act = {"sin": SinAct(), "sinc": SincAct()}[act_type]

    def forward(self, x):
        x = self.omega * self.linear(x)
        x = self.act(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, batchnorms=True, act_type="silu", shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=not batchnorms)
        self.norm = torch.nn.BatchNorm1d(dim_out) if batchnorms else torch.nn.Identity()
        self.act = {
            "relu": torch.nn.ReLU(inplace=True),
            "silu": torch.nn.SiLU(inplace=True),
            "sin": SinAct(),
        }[act_type]
        if shortcut:
            if dim_in != dim_out:
                raise ValueError("Invalid layer configuration")
            self.weight = torch.nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        shortcut = x
        x = self.linear(x)
        x = self.norm(x)
        if self.shortcut:
            x = self.weight * x + shortcut
        x = self.act(x)
        return x


class SirenWithResblock(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        sirenblock_is_first=False,
        sirenblock_omega=30,
        sirenblock_act_type="sinc",
        resblock_batchnorms=True,
        resblock_act_type="silu",
        resblock_shortcut=True,
        resblock_channels_fraction=0.5,
    ):
        super().__init__()
        dim_out_resblock = max(int(resblock_channels_fraction * dim_out), 1)
        dim_out_siren = dim_out - dim_out_resblock
        self.siren = SirenLayer(
            dim_in, dim_out_siren,
            is_first=sirenblock_is_first,
            omega=sirenblock_omega,
            act_type=sirenblock_act_type,
        )
        self.residual = ResBlock(
            dim_in, dim_out_resblock,
            batchnorms=resblock_batchnorms,
            act_type=resblock_act_type,
            shortcut=resblock_shortcut,
        )

    def forward(self, x):
        return torch.cat((self.siren(x), self.residual(x)), dim=-1)
