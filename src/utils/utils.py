import torch
from hydra.utils import instantiate

from .config import get_prev_config


def load_model(path, model_cls=None, device=None):

    cfg = get_prev_config(path)

    # TODO: tidy this to just use instantiate
    model = model_cls(cfg) if model_cls is not None else instantiate(cfg.model, cfg=cfg)
    # model = instantiate(cfg.model, cfg=cfg)
    sdict = torch.load(path + "/model.pt", weights_only=False, map_location='cpu')
    model.load_state_dict(sdict["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if device is not None:
        model = model.to(device)

    # model = torch.compile(model)

    return model, cfg