import torch
from lgatr import get_spurions

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
