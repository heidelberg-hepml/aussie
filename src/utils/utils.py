import torch
import numpy as np
from hydra.utils import instantiate

from .config import get_prev_config


def load_model(path, model_cls=None, device=None, freeze=True):

    cfg = get_prev_config(path)

    # TODO: tidy this to just use instantiate
    model = model_cls(cfg) if model_cls is not None else instantiate(cfg.model, cfg=cfg)
    # model = instantiate(cfg.model, cfg=cfg)
    sdict = torch.load(path + "/model.pt", weights_only=False, map_location="cpu")

    # load (and optionally freeze) weights
    model.load_state_dict(sdict["model"])
    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    # set to eval mode (disable batchnorm, dropout etc.)
    model.eval()

    # move to device
    if device is not None:
        model = model.to(device)

    return model, cfg


def norm_squared(p, keepdims=False):
    return np.sum(p[..., 1:] ** 2, axis=-1, keepdims=keepdims)


def norm(p, keepdims=False):
    return np.sqrt(norm_squared(p, keepdims=keepdims))


def mass_squared(p):
    return p[..., 0] ** 2 - np.sum(p[..., 1:] ** 2, axis=-1)


def mass(p, clip=True):
    mass_2 = mass_squared(p)
    if len(mass_2[mass_2 < 0]) > 0 and 1 < 0:
        print("Mass array shape:", mass_2.shape)
        print("Mass array negative masses ratio:", (mass_2 < 0).mean())
        print("Mass array negative masses ratio <-0.1:", (mass_2 < -0.1).mean())
        print("Mass array negative masses ratio <-1:", (mass_2 < -1).mean())
        print("Mass array negative masses mean:", mass_2[mass_2 < 0].mean(0))
        print("Mass array negative masses min:", mass_2[mass_2 < 0].min(0))
        print("Mass array negative masses std:", mass_2[mass_2 < 0].std(0))

    if clip:
        return np.sqrt(np.clip(mass_2, a_min=0, a_max=None))
    else:
        return np.sqrt(mass_2)


def pt2(p):
    return p[..., 1] ** 2 + p[..., 2] ** 2


def pt(p):
    return np.sqrt(pt2(p))


def eta(p):
    # p_norm = np.sqrt(np.sum(p[...,1:]**2, axis=-1))
    # return 0.5 * np.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))
    return np.arctanh(
        p[..., 3] / np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2 + p[..., 3] ** 2)
    )


def azimuthal_angle(p):
    return np.arctan2(p[..., 2], p[..., 1])


def cartesian2jet(p_eppp):

    p = np.full(p_eppp.shape, np.nan)
    p[..., 0] = mass(p_eppp)
    p[..., 1] = pt(p_eppp)
    p[..., 2] = eta(p_eppp)
    p[..., 3] = azimuthal_angle(p_eppp)

    return p
