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
        self.bayesian = self.head.bayesian
        self.out_norm = nn.LayerNorm(head.dim_in)
        self.attn_context = lambda: (
            nullcontext() if backbone.compile else sdpa_kernel(SDPBackend.MATH)
        )

        self.register_buffer(
            "spurions",
            get_spurions(beam_spurion="lightlike")[None, ..., 1:5],
        )
        self.num_spurions = self.spurions.size(1)
        # self.register_buffer(
        #     "time_reference", torch.tensor([[[1, 0, 0, 0]]], dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "beam_reference", torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float32)
        # )

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

    # @property
    # def kld(self):
    #     if self.bayesian:
    #         return self.head.kld

    # def reseed(self):
    #     if self.bayesian:
    #         self.head.reseed()


class LGATrClass(LGATr):

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: Optional[torch.nn.Module] = None,
        # encode_pos: bool = False,
    ):

        super().__init__(backbone, head)
        # self.register_buffer(
        #     "global_token_v", torch.tensor([[[0, 0, 0, 0]]], dtype=torch.float32)
        # )
        self.register_buffer(
            "spurions",
            get_spurions(beam_spurion="lightlike")[None, ..., 1:5],
        )
        self.N_extra = self.spurions.size(1) + 1

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        v, s = x[..., :4], x[..., 4:]

        B, N, C = s.shape

        # add time and beam references
        v = torch.cat(
            [
                # self.global_token_v.expand(B, -1, -1),
                torch.zeros(B, 1, 4, device=x.device, dtype=x.dtype),
                self.spurions.expand(B, -1, -1),
                # self.time_reference.expand(B, -1, -1),
                # self.beam_reference.expand(B, -1, -1),
                v,
            ],
            dim=1,
        )
        s = torch.cat(
            [torch.zeros(B, self.N_extra, C, device=x.device, dtype=x.dtype), s],
            dim=1,
        )

        if mask is None:
            mask = torch.ones((B, N + self.N_extra), device=x.device, dtype=torch.bool)
        else:
            mask = F.pad(mask, (self.N_extra, 0), value=1)

        v = v.unsqueeze(2)

        # forward lgatr backbone
        with self.attn_context():
            _, z = self.backbone(v, s, attn_mask=mask.view(B, 1, 1, N + self.N_extra))

        return z[:, 0].mean(-1, keepdims=True)


class LGATrChanref(LGATr):

    def __init__(
        self,
        backbone: torch.nn.Module,
        head: Optional[torch.nn.Module] = None,
        # encode_pos: bool = False,
    ):

        super().__init__(backbone, head)
        self.register_buffer(
            "spurions",
            get_spurions(beam_spurion="lightlike")[None, None, :, 1:5],  # (1, 1, 3, 4)
        )
        # self.register_buffer(
        #     "time_reference", torch.tensor([[[[1, 0, 0, 0]]]], dtype=torch.float32)
        # )
        # self.register_buffer(
        #     "beam_reference", torch.tensor([[[[0, 0, 0, 1]]]], dtype=torch.float32)
        # )

    def forward(
        self,
        x: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, N, _ = x.shape

        # split into vector and scalar components
        v, s = x[..., :4], x[..., 4:]
        v = v.unsqueeze(2)

        # add time and beam references
        v = torch.cat(
            [
                # self.time_reference.expand(B, N, -1, -1),
                # self.beam_reference.expand(B, N, -1, -1),
                self.spurions.expand(B, N, -1, -1),
                v,
            ],
            dim=-2,
        )  # (B, N, 1+3, 4)

        if mask is None:
            mask = torch.ones((B, N), device=x.device, dtype=torch.bool)

        # forward lgatr backbone
        with self.attn_context():
            _, z = self.backbone(v, s, attn_mask=mask.view(B, 1, 1, N))

        # aggregate
        z[~mask] = 0.0
        z = z.sum(1) / mask.sum(1, keepdim=True)

        # norm and project output
        z = self.out_norm(z)
        if self.head is not None:
            z = self.head(z)

        return z
