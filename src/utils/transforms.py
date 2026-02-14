import torch
from tensordict import tensorclass
from typing import Optional
from lgatr import get_spurions
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
        x[..., indices] = x[..., indices].add(eps).log()
        return x

    @staticmethod
    def reverse(x, indices):
        x[..., indices] = x[..., indices].exp()
        return x


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


def lorentz_inner(v1, v2):
    """Lorentz inner product, i.e v1^T @ g @ v2

    Parameters
    ----------
    v1, v2 : torch.Tensor
        Tensors of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz inner product of shape (..., )
    """
    t = v1[..., 0] * v2[..., 0]
    s = (v1[..., 1:] * v2[..., 1:]).sum(dim=-1)
    return t - s


def lorentz_squarednorm(v):
    """Lorentz norm, i.e. v^T @ g @ v

    Parameters
    ----------
    v : torch.Tensor
        Tensor of shape (..., 4)

    Returns
    -------
    torch.Tensor
        Lorentz norm of shape (..., )
    """
    return lorentz_inner(v, v)


def restframe_boost(fourmomenta, checks=False):
    """Construct a Lorentz transformation that boosts four-momenta into their rest frame.

    Parameters
    ----------
    fourmomenta : torch.Tensor
        Tensor of shape (..., 4) representing the four-momenta.
    checks : bool
        If True, perform additional assertion checks on predicted vectors.
        It may cause slowdowns due to GPU/CPU synchronization, use only for debugging.

    Returns
    -------
    trafo : torch.Tensor
        Tensor of shape (..., 4, 4) representing the Lorentz transformation
        that boosts the four-momenta into their rest frame.
    """
    if checks:
        assert (
            lorentz_squarednorm(fourmomenta) > 0
        ).all(), "Trying to boost spacelike vectors into their restframe (not possible). Consider changing the nonlinearity in equivectors."

    # compute relevant quantities
    t0 = fourmomenta.narrow(-1, 0, 1)
    beta = fourmomenta[..., 1:] / t0.clamp_min(1e-10)
    beta2 = beta.square().sum(dim=-1, keepdim=True)
    one_minus_beta2 = torch.clamp_min(1 - beta2, min=1e-10)
    gamma = torch.rsqrt(one_minus_beta2)
    boost = -gamma * beta

    # prepare rotation part
    eye3 = torch.eye(3, device=fourmomenta.device, dtype=fourmomenta.dtype)
    eye3 = eye3.reshape(*(1,) * len(fourmomenta.shape[:-1]), 3, 3).expand(
        *fourmomenta.shape[:-1], 3, 3
    )
    scale = (gamma - 1) / torch.clamp_min(beta2, min=1e-10)
    outer = beta.unsqueeze(-1) * beta.unsqueeze(-2)
    rot = eye3 + scale.unsqueeze(-1) * outer

    # collect trafo
    row0 = torch.cat((gamma, boost), dim=-1)
    lower = torch.cat((boost.unsqueeze(-1), rot), dim=-1)
    trafo = torch.cat((row0.unsqueeze(-2), lower), dim=-2)
    return trafo


class LorentzTransform:
    """
    TODO
    """

    @staticmethod
    def forward(features, boost_momenta=None, mask=None, eps=1e-4):

        B = len(features)

        vectors, scalars = features[..., :4], features[..., 4:]

        # create reference features
        ref_vectors = get_spurions(beam_spurion="lightlike")[..., 1:5]  # (3, 4)
        ref_scalars = -torch.ones_like(scalars[:, [0]])  # (B, 1, C)

        # concat with particle features
        vectors = torch.cat([ref_vectors[None, ...].expand(B, -1, -1), vectors], dim=1)
        scalars = torch.cat([ref_scalars.expand(-1, 3, -1), scalars], dim=1)

        if boost_momenta is not None:
            # boost vectors to rest frame
            boost = restframe_boost(boost_momenta)
            vectors = torch.einsum("ijk,ilk->ilj", boost, vectors)

        # concat with particle features
        features = torch.cat([vectors, scalars], dim=2)

        if mask is not None:
            # update mask
            mask = torch.nn.functional.pad(mask, (3, 0), value=1)

        return features, mask

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


@tensorclass
class OmniFoldObsTransform:
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
        batch.x[:, 4] = 1 - batch.x[:, 4]
        batch.x = LogScale.forward(batch.x, indices=[0, 1, 3, 4], eps=self.eps)
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        batch.z[:, 4] = 1 - batch.z[:, 4]
        batch.z = LogScale.forward(batch.z, indices=[0, 1, 3, 4], eps=self.eps)
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x = LogScale.reverse(batch.x, indices=[0, 1, 3, 4], eps=self.eps)
        batch.x[:, 4] = 1 - batch.x[:, 4]

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z = LogScale.reverse(batch.x, indices=[0, 1, 3, 4], eps=self.eps)
        batch.z[:, 4] = 1 - batch.z[:, 4]

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


@tensorclass
class YukawaEquiTransform:
    """
    TODO
    """

    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-4

    def forward(self, batch):

        batch.x /= self.scale_x
        batch.z /= self.scale_z

        # create particle ids
        ids_x = torch.eye(4, device=batch.device)[
            None, [0, 1, 2, 3, 3, 3, 3, 3], :
        ].expand(len(batch), -1, -1)
        ids_z = torch.eye(3, device=batch.device)[None, [0, 1, 2], :].expand(
            len(batch), -1, -1
        )
        batch.x = torch.cat([batch.x, ids_x], dim=2)
        batch.z = torch.cat([batch.z, ids_z], dim=2)

        batch.x, batch.mask_x = LorentzTransform.forward(batch.x, mask=batch.mask_x)
        batch.z, batch.mask_z = LorentzTransform.forward(batch.z, mask=batch.mask_z)

        return batch

    def reverse(self, batch):
        raise NotImplementedError
        return batch


@tensorclass
class YukawaAcceptanceTransform:
    """
    TODO
    """

    scale: torch.Tensor
    eps: float = 1e-4

    def forward(self, batch):

        batch.x /= self.scale

        # create particle ids
        ids = torch.eye(3, device=batch.device)[None, [0, 1, 2], :].expand(
            len(batch), -1, -1
        )
        batch.x = torch.cat([batch.x, ids], dim=2)

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


@tensorclass
class OmniFoldCartesianTransform:
    """
    TODO
    """

    def forward(self, batch):

        # # scale down vectors
        # batch.x[..., :4] = batch.x[..., :4].divide(self.scale_v)
        # batch.z[..., :4] = batch.z[..., :4].divide(self.scale_v)

        batch.x, batch.mask_x = LorentzTransform.forward(
            batch.x, batch.cond_x, batch.mask_x
        )
        batch.z, batch.mask_z = LorentzTransform.forward(
            batch.z, batch.cond_z, batch.mask_z
        )

        return batch

    def reverse(self, batch):

        batch.x, batch.mask_x = LorentzTransform.reverse(
            batch.x, batch.cond_x, batch.mask_x
        )
        batch.z, batch.mask_z = LorentzTransform.reverse(
            batch.z, batch.cond_z, batch.mask_z
        )

        return batch
