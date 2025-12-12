import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from itertools import pairwise
from typing import Optional

from .layers import BayesianLinear, StackedLinear


class MLP(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_channels: int = 64,
        num_hidden_layers: int = 1,
        act: str = "relu",
        out_act: Optional[str] = None,
        drop: float = 0.0,
        bayesian: bool = False,
        bayesian_prior_prec: float = 1.0,
        init_gain: float = 1.0,
        out_bias_gain: float = 1.0,
        ensembled: Optional[int] = None,
        layernorms: bool = False,
    ):

        super().__init__()

        units = [dim_in, *(num_hidden_layers * [hidden_channels]), dim_out]

        self.bayesian = bayesian
        self.ensembled = ensembled
        linear_layer = (
            partial(BayesianLinear, prior_prec=bayesian_prior_prec)
            if bayesian
            else (
                partial(StackedLinear, channels=ensembled, gain=init_gain)
                if ensembled
                else nn.Linear
            )
        )
        self.linear_layers = nn.ModuleList(
            [linear_layer(a, b) for a, b in pairwise(units)]
        )

        self.layernorms = layernorms
        if layernorms:
            self.norm_layers = nn.ModuleList(
                [nn.LayerNorm(hidden_channels) for _ in range(num_hidden_layers)]
            )

        # if ensembled:
        #     nn.init.uniform_(self.linear_layers[-1].bias, -out_bias_gain, out_bias_gain)

        self.act = getattr(F, act)
        self.out_act = getattr(F, out_act) if out_act else None
        self.drop = nn.Dropout(drop) if drop else None

    def forward(self, x):

        if self.ensembled:
            # repeat input in ensemble dimension
            x = x.expand(self.ensembled, -1, -1)

        for i, linear in enumerate(self.linear_layers[:-1]):

            x = linear(x)
            x = self.act(x)
            if self.drop is not None:
                x = self.drop(x)
            if self.layernorms:
                x = self.norm_layers[i](x)

        x = self.linear_layers[-1](x)
        if self.out_act is not None:
            x = self.out_act(x)

        return x

    @property
    def kld(self):
        if self.bayesian:
            return sum(layer.kld for layer in self.linear_layers)

    def reseed(self):
        if self.bayesian:
            for layer in self.linear_layers:
                layer.reseed()
