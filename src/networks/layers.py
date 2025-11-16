import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from typing import Optional


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()

        inter_dim = int(dim * mult)
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        is_causal: bool = False,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.is_causal = is_causal
        self.drop_attn = nn.Dropout(drop_attn)
        self.proj = nn.Linear(dim, dim)
        self.drop_proj = nn.Dropout(drop_proj)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # dont attend to padded elements
        attn_mask = None if mask is None else mask.view(B, 1, 1, N)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.is_causal,
            attn_mask=attn_mask,
            dropout_p=self.drop_attn.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.drop_proj(x)
        return x


class BayesianLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_prec: float = 1.0,
        _map: bool = False,
        std_init: float = -9,
    ):
        """
        :param in_features: input dimension
        :param out_features: output dimension
        :param prior_prec: 1/sigma_prior^2, width of prior weight distribution
        :param _map: if true weights will not be sampled but maximum-a-posteriori (mean) will be used instead
        :param std_init: initialization of learned sigma
        """
        super().__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.map = _map
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.empty(out_features))
        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.empty(out_features, in_features))
        self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(self.std_init, 0.001)
        self.bias.data.zero_()

    @property
    def kld(self):
        # limit values to -11, 11
        logsig2_w = self.logsig2_w.clamp(-11, 11)
        # KL divergence of two gaussians
        kld = (
            self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
            - logsig2_w
            - 1
            - np.log(self.prior_prec)
        ).sum() / 2
        return kld

    def forward(self, input):
        if self.training:
            # local reparameterization trick is more efficient and leads to
            # an estimate of the gradient with smaller variance.
            # https://arxiv.org/pdf/1506.02557.pdf
            mu_out = F.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            if self.map:  # just return the mean, no sampling
                return F.linear(input, self.mu_w, self.bias)

            logsig2_w = self.logsig2_w.clamp(-11, 11)
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            # sample gaussian random numbers
            weight = self.mu_w + s2_w.sqrt() * self.random
            return F.linear(input, weight, self.bias) + 1e-8
        
    def reseed(self):
        self.random = None