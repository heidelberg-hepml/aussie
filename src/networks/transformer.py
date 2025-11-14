import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from typing import Optional

from .layers import Attention, BayesianLinear, FeedForward


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_channels: int,
        num_blocks: int,
        num_heads: int,
        mlp_ratio: float,
        dim_cond: Optional[int] = None,
        checkpoint_grads: bool = False,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        drop_mlp: float = 0.0,
        bayesian: bool = False,
        max_len: int = 100,
        pos_dim: int = 5

    ):

        super().__init__()
        self.dim_in = dim_in
        self.bayesian = bayesian
        self.max_len = max_len
        # input/output embeddings
        self.proj_in = nn.Linear(dim_in, hidden_channels)
        self.proj_out = (BayesianLinear if bayesian else nn.Linear)(
            hidden_channels, dim_out
        )

        # init condition embedding if needed to bridge dimensions
        self.conditional = dim_cond is not None
        if self.conditional:
            self.proj_cond = nn.Linear(dim_cond, hidden_channels)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    checkpoint_grads=checkpoint_grads,
                    drop_mlp=drop_mlp,
                    drop_attn=drop_attn,
                    drop_proj=drop_proj,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        # self.positional_encoding = nn.Embedding(max_len, pos_dim)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # one_hot = torch.eye(self.max_len, device=x.device, dtype=x.dtype)[
        #           None, : x.shape[1], :
        #           ].expand(x.shape[0], -1, -1)
        # x = torch.cat([x, one_hot], dim=-1)
        # batch_size, seq_len, _ = x.size()
        # pos_enc = self.positional_encoding(torch.arange(seq_len, device=x.device))
        # pos_enc = pos_enc.unsqueeze(0)
        # pos_enc = pos_enc.expand(batch_size, -1, -1)
        # if pos_enc.size(-1) != x.size(-1):
        #     pos_enc = pos_enc.unsqueeze(-1).expand(-1, -1, x.size(-1))
        #
        # x = x + pos_enc
        x = self.proj_in(x)


        if self.conditional:
            # append condition as token
            c = self.proj_cond(c).unsqueeze(1)
            x = torch.cat([c, x], dim=1)
            mask = F.pad(mask, (1, 0), value=1)

        # forward pass through transformer stack
        for block in self.blocks:
            x = block(x, mask=mask)

        # aggregate
        x[~mask] = 0
        x = x.sum(1)

        # norm and project output
        x = self.out_norm(x)
        x = self.proj_out(x)

        return x

    @property
    def kld(self):
        if self.bayesian:
            return self.proj_out.kld

    def reseed(self):
        if self.bayesian:
            self.proj_out.reseed()


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        is_causal: bool = False,
        checkpoint_grads: bool = False,
        drop_mlp: float = 0.0,
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attn = Attention(
            hidden_channels,
            num_heads=num_heads,
            qkv_bias=True,
            is_causal=is_causal,
            drop_attn=drop_attn,
            drop_proj=drop_proj,
        )
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffwd = FeedForward(dim=hidden_channels, mult=mlp_ratio, dropout=drop_mlp)
        self.checkpoint_grads = checkpoint_grads

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        x = self.norm1(x)
        if self.checkpoint_grads:
            x = x + checkpoint(self.attn, x, mask, use_reentrant=False)
        else:
            x = x + self.attn(x, mask)
        x = x + self.ffwd(self.norm2(x))

        return x
