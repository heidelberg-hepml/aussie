import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from typing import Optional
from xformers.ops.fmha import BlockDiagonalMask

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
        pos_dim: int = 5,
        head: Optional[torch.nn.Module] = None,
        use_jvp: bool = False,
        encode_pos: bool = False,
    ):

        super().__init__()
        self.dim_in = dim_in
        self.bayesian = bayesian

        self.max_len = max_len
        # input/output embeddings
        self.proj_in = nn.Linear(dim_in, hidden_channels)
        self.head = head
        self.use_jvp = use_jvp
        self.encode_pos = encode_pos

        # self.ensembled = False  # TODO: Implement parallel ensembling
        self.ensembled = self.head.ensembled

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
                    use_jvp=use_jvp,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        if self.encode_pos:
            self.pos_encoding = nn.Embedding(max_len, hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if mask is None:
            mask = torch.ones_like(x[..., 0], dtype=torch.bool)

        x = self.proj_in(x)
        if self.encode_pos:
            x = x + self.pos_encoding.weight[None, : x.size(1), :]

        if self.conditional:
            # append condition as token
            c = self.proj_cond(c).unsqueeze(1)
            x = torch.cat([c, x], dim=1)
            mask = F.pad(mask, (1, 0), value=1)
        elif c is not None:
            raise RuntimeError(
                "Received condition in `TransformerEncoder.forward`, but `dim_cond` not provided at initialization"
            )

        lengths = mask.int().sum(1)
        if not self.use_jvp:
            # create packed tensor
            x = x[None, ..., mask, :]
            mask = BlockDiagonalMask.from_seqlens(lengths.tolist(), device=x.device)

        # forward pass through transformer stack
        for block in self.blocks:
            x = block(x, mask=mask)

        # aggregate
        if self.use_jvp:
            x[~mask] = 0.0
            x = x.sum(1) / lengths[..., None]
        else:
            x = torch.segment_reduce(x.squeeze(0), "mean", lengths=lengths)

        # norm and project output
        x = self.out_norm(x)

        if self.head is not None:
            x = self.head(x)

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
        use_jvp: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.attn = Attention(
            hidden_channels,
            num_heads=num_heads,
            qkv_bias=True,
            is_causal=is_causal,
            drop_attn=drop_attn,
            drop_proj=drop_proj,
            use_jvp=use_jvp,
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
