import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional
from lgatr import get_spurions


class LGATr(nn.Module):

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: Optional[torch.nn.Module] = None,
        # encode_pos: bool = False,
    ):

        super().__init__()
        self.backbone = backbone
        self.head = head
        self.ensembled = self.head.ensembled
        self.out_norm = nn.LayerNorm(head.dim_in)
        self.attn_context = lambda: (
            nullcontext() if backbone.compile else sdpa_kernel(SDPBackend.MATH)
        )

        self.lowlevel = True

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        v, s = x[..., :4], x[..., 4:]

        B, N, C = s.shape

        if mask is None:
            mask = torch.ones((B, N), device=x.device, dtype=torch.bool)

        v = v.unsqueeze(2)

        # forward lgatr backbone
        with self.attn_context():
            _, z = self.backbone(v, s, attn_mask=mask.view(B, 1, 1, N))

        # aggregate
        z = (z * mask.unsqueeze(-1).int()).sum(1)  # / mask.sum(1, keepdim=True)

        # norm and project output
        z = self.out_norm(z)
        if self.head is not None:
            z = self.head(z)

        return z
