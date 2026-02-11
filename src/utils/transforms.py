import torch
from tensordict import tensorclass
from typing import Optional

from src.utils.utils import cartesian2jet


class ShiftAndScale:
    """
    x -> (x - shift) / scale
    """

    @staticmethod
    def forward(x, shift=0, scale=1):
        return x.sub(shift).div(scale)

    @staticmethod
    def reverse(x, shift=0, scale=1):
        return x.multiply(scale).add(shift)


class LogScale:
    """
    x -> log(x) + eps
    """

    @staticmethod
    def forward(x, indices, eps=1e-6):
        return x[..., indices].add(eps).log()

    @staticmethod
    def reverse(x, indices):
        return x[..., indices].exp()


class MomentumTransform:
    """
    Convert (E, px, py, pz) to (log m^2, log pt^2, eta, phi)
    """

    @staticmethod
    def forward(mom, eps=1e-4):

        E, px, py, pz = mom.unbind(-1)

        pt2 = px**2 + py**2
        p2 = pt2 + pz**2
        phi = torch.arctan2(py, px)
        eta = torch.arctanh(pz / (p2.sqrt() + eps))
        m2 = (E**2 - p2).clamp(min=0)

        return torch.stack([m2.add(eps).log(), pt2.add(eps).log(), eta, phi], dim=-1)

    @staticmethod
    def reverse(x):
        raise NotImplementedError


# TODO: split into Transformx and Transformz
@tensorclass
class OmniFoldTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # process reco level
        batch.x[:, 1] += torch.rand_like(batch.x[:, 1]) - 0.5
        batch.x[:, :4] = batch.x[:, :4].add(self.eps).log()
        # batch.x[:, [0, 1, 2, 3, 6]] = batch.x[:, [0, 1, 2, 3, 6]].add(self.eps).log()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        batch.z[:, 1] += torch.rand_like(batch.z[:, 1]) - 0.5
        batch.z[:, :4] = batch.z[:, :4].add(self.eps).log()
        # batch.z[:, [0, 1, 2, 3, 6]] = batch.z[:, [0, 1, 2, 3, 6]].add(self.eps).log()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[:, :4] = batch.x[:, :4].exp().sub(self.eps)
        batch.x[:, 1] = batch.x[:, 1].round()

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[:, :4] = batch.z[:, :4].exp().sub(self.eps)
        batch.z[:, 1] = batch.z[:, 1].round()

        return batch


# TODO: split into Transformx and Transformz
@tensorclass
class OmniFoldParticleTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    shift_cond_x: Optional[torch.Tensor] = None
    shift_cond_z: Optional[torch.Tensor] = None
    scale_cond_x: Optional[torch.Tensor] = None
    scale_cond_z: Optional[torch.Tensor] = None
    eps: float = 1e-8

    def forward(self, batch):

        # process reco level
        # fmt: off
        batch.x[..., 0] = batch.x[..., 0].multiply(100).add(self.eps).log()  # log-scale pT
        batch.z[..., 0] = batch.z[..., 0].multiply(100).add(self.eps).log()

        if batch.cond_x is not None:  # jet kinematics
            
            batch.x[..., 0] -= batch.cond_x[:, [0]].log()  # use relative pT for constituents
            batch.z[..., 0] -= batch.cond_z[:, [0]].log()           
            batch.cond_x[..., [0, 3]] = batch.cond_x[..., [0, 3]].add(1e-3).log()  # log-scale pT, m
            batch.cond_z[..., [0, 3]] = batch.cond_z[..., [0, 3]].add(1e-3).log()
            batch.cond_x = ShiftAndScale.forward(batch.cond_x, shift=self.shift_cond_x, scale=self.scale_cond_x)
            batch.cond_z = ShiftAndScale.forward(batch.cond_z, shift=self.shift_cond_z, scale=self.scale_cond_z)
        
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)
        # fmt: on

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[:, :4] = batch.x[:, :4].exp().sub(self.eps)
        batch.x[:, 1] = batch.x[:, 1].round()

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[:, :4] = batch.z[:, :4].exp().sub(self.eps)
        batch.z[:, 1] = batch.z[:, 1].round()

        return batch


@tensorclass
class OmniFoldParticleTransform2:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    shift_cond_x: Optional[torch.Tensor] = None
    shift_cond_z: Optional[torch.Tensor] = None
    scale_cond_x: Optional[torch.Tensor] = None
    scale_cond_z: Optional[torch.Tensor] = None
    eps: float = 1e-8

    def forward(self, batch):

        # process reco level
        # fmt: off
        batch.x[..., 0] = batch.x[..., 0].multiply(100).add(self.eps).log()  # log-scale pT
        batch.z[..., 0] = batch.z[..., 0].multiply(100).add(self.eps).log()

        if batch.cond_x is not None:  # jet kinematics
            
            batch.x[..., 0] -= batch.cond_x[:, [0]].log()  # use relative pT for constituents
            batch.z[..., 0] -= batch.cond_z[:, [0]].log()
            batch.cond_x[..., [0, 3]] = batch.cond_x[..., [0, 3]].add(1e-3).log()  # log-scale pT, m
            batch.cond_z[..., [0, 3]] = batch.cond_z[..., [0, 3]].add(1e-3).log()

            # keep only pt, eta in condition
            batch.cond_x = batch.cond_x[..., [0,1]]
            batch.cond_z = batch.cond_z[..., [0,1]]

            batch.cond_x = ShiftAndScale.forward(batch.cond_x, shift=self.shift_cond_x, scale=self.scale_cond_x)
            batch.cond_z = ShiftAndScale.forward(batch.cond_z, shift=self.shift_cond_z, scale=self.scale_cond_z)
        
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)
        # fmt: on

        return batch

    def reverse(self, batch):

        raise NotImplementedError

        return batch


@tensorclass
class ttbarTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # process reco level
        Ms_x = [
            compute_invariant_mass(batch.x, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]
        batch.x = batch.x.flatten(-2, -1)
        batch.x = torch.cat([batch.x, *Ms_x], dim=1)
        batch.x[..., [0, 1, 4, 5, 8, 9]] = (
            batch.x[..., [0, 1, 4, 5, 8, 9]].add(self.eps).log()
        )
        # batch.x[..., [12, 13, 14]] -= 80.3
        # batch.x[..., 15] -= 172.5
        # batch.x[..., [12, 13, 14, 15]] = batch.x[..., [12, 13, 14, 15]].arcsinh()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x = torch.cat([batch.x, batch.conds.unsqueeze(1)], dim=1)

        # process part level
        Ms_z = [
            compute_invariant_mass(batch.z, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]
        batch.z = batch.z.flatten(-2, -1)
        batch.z = torch.cat([batch.z, *Ms_z], dim=1)
        batch.z[..., [0, 1, 4, 5, 8, 9]] = (
            batch.z[..., [0, 1, 4, 5, 8, 9]].add(self.eps).log()
        )
        # batch.z[..., [12, 13, 14]] -= 80.3
        # batch.z[..., 15] -= 172.5
        # batch.z[..., [12, 13, 14, 15]] = batch.z[..., [12, 13, 14, 15]].arcsinh()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z = torch.cat([batch.z, batch.conds.unsqueeze(1)], dim=1)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = batch.x.unflatten(-1, (3, 4))
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[..., :2] = batch.x[..., :2].exp().sub(self.eps)

        # process part level
        batch.z = batch.z.unflatten(-1, (3, 4))
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[..., :2] = batch.z[..., :2].exp().sub(self.eps)

        return batch


@tensorclass
class ttbarParticleTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    shift_cond_x: torch.Tensor
    shift_cond_z: torch.Tensor
    scale_cond_x: torch.Tensor
    scale_cond_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # virtual masses
        Ms_x = [
            compute_invariant_mass(batch.x, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]
        Ms_z = [
            compute_invariant_mass(batch.z, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2], [0, 1, 2])
        ]

        # log-scale pt and mass
        batch.x[..., [0, 1]] = batch.x[..., [0, 1]].add(self.eps).log()
        batch.z[..., [0, 1]] = batch.z[..., [0, 1]].add(self.eps).log()
        # normalize
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        # combine virtual mass with sim mass label
        batch.cond_x = torch.cat([*Ms_x, batch.conds.unsqueeze(1)], dim=1)
        batch.cond_z = torch.cat([*Ms_z, batch.conds.unsqueeze(1)], dim=1)
        batch.cond_x = ShiftAndScale.forward(
            batch.cond_x, shift=self.shift_cond_x, scale=self.scale_cond_x
        )
        batch.cond_z = ShiftAndScale.forward(
            batch.cond_z, shift=self.shift_cond_z, scale=self.scale_cond_z
        )

        return batch

    def reverse(self, batch):
        raise NotImplementedError
        return batch


@tensorclass
class YukawaTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-4

    def forward(self, batch):

        # transform to (log m^2, log pt^2, eta, phi)
        batch.x = MomentumTransform.forward(batch.x, eps=self.eps)
        batch.z = MomentumTransform.forward(batch.z, eps=1e-3)

        # drop mass for on-shell part level
        batch.z = batch.z[..., 1:]

        # normalize
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):
        raise NotImplementedError
        return batch


def compute_invariant_mass(p, particles) -> torch.Tensor:

    px_sum = 0
    py_sum = 0
    pz_sum = 0
    e_sum = 0
    for particle in particles:

        m = p[..., particle, 0]
        pT = p[..., particle, 1]
        eta = p[..., particle, 2]
        phi = p[..., particle, 3]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    m = torch.sqrt(
        torch.clamp((e_sum) ** 2 - (px_sum) ** 2 - (py_sum) ** 2 - (pz_sum) ** 2, min=0)
    )
    return m


@tensorclass
class ttbarRawTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # process reco level
        Ms_x = [
            compute_invariant_mass(batch.x, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2])
        ]
        batch.x = batch.x.flatten(-2, -1)
        batch.x = torch.cat([batch.x[..., [0, 4, 8]], *Ms_x], dim=1)
        batch.x[..., :3] = batch.x[..., :3].add(self.eps).log()  # log masses
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x = torch.cat([batch.x, batch.conds.unsqueeze(1)], dim=1)

        # process part level
        Ms_z = [
            compute_invariant_mass(batch.z, ps).unsqueeze(1)
            for ps in ([0, 1], [0, 2], [1, 2])
        ]
        batch.z = batch.z.flatten(-2, -1)
        batch.z = torch.cat([batch.z[..., [0, 4, 8]], *Ms_z], dim=1)
        batch.z[..., :3] = batch.z[..., :3].add(self.eps).log()  # log masses§
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z = torch.cat([batch.z, batch.conds.unsqueeze(1)], dim=1)

        return batch

    def reverse(self, batch):

        raise NotImplementedError

        return batch
