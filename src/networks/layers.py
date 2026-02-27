import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from xformers.ops import memory_efficient_attention
from jvp_flash_attention.jvp_attention import attention as jvp_attention
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
        use_jvp: bool = False,
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
        self.use_jvp = use_jvp

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.use_jvp:

            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            # # dont attend to padded elements
            attn_mask = (
                None
                if mask is None
                else mask.view(B, 1, 1, N).expand(-1, self.num_heads, N, -1)
            )

            x = jvp_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                causal=self.is_causal,
                dropout_p=self.drop_attn.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # x = F.scaled_dot_product_attention(
            #     q,
            #     k,
            #     v,
            #     is_causal=self.is_causal,
            #     attn_mask=attn_mask,
            #     dropout_p=self.drop_attn.p if self.training else 0.0,
            # )

            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 1, 3, 4)
            )
            q, k, v = qkv.unbind(0)
            q, k = self.q_norm(q), self.k_norm(k)

            x = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=mask,
                p=self.drop_attn.p if self.training else 0.0,
            ).flatten(2, 3)

        x = self.proj(x)
        x = self.drop_proj(x)
        return x


class StackedLinear(nn.Module):
    "Efficient implementation of linear layers for ensembles of networks"

    def __init__(self, in_features, out_features, channels, gain=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels
        self.gain = gain
        self.weight = nn.Parameter(torch.empty((channels, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((channels, out_features)))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.channels):

            # # orig
            torch.nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias[i], -bound, bound)
            # xavier
            # torch.nn.init.xavier_uniform_(self.weight[i], gain=self.gain) # new
            # torch.nn.init.xavier_uniform_(self.bias[i], gain=self.gain, generator=None)

    def forward(self, input):
        return torch.baddbmm(self.bias[:, None, :], input, self.weight.transpose(1, 2))
