import os
import torch
import torch.nn as nn
import logging

from abc import abstractmethod
from collections import defaultdict
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.datasets import UnfoldingData

log = logging.getLogger("Model")


class Model(nn.Module):

    def __init__(self, cfg: DictConfig):

        super().__init__()

        self.cfg = cfg

        # networks
        self.net = instantiate(cfg.net)

        # logging
        self.log_buffer = defaultdict(list)

        # ensembling
        self.ensembled = self.net.ensembled

    @abstractmethod
    def batch_loss(self, batch: UnfoldingData, training=True):
        pass

    @property
    def trainable_parameters(self):
        return (p for p in self.parameters() if (p.requires_grad and p.numel() > 0))

    def update(self, loss, optimizer, scaler, step=None, total_steps=None):

        # zero parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # scale gradients for mixed precision stability
        loss = scaler.scale(loss)

        # propagate gradients
        loss.backward()

        # optionally clip gradients
        if clip := self.cfg.training.gradient_norm:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.trainable_parameters, clip)
            self.log_scalar(grad_norm, "gradient_norm")

        # update weights
        scaler.step(optimizer)
        scaler.update()

        # # zero parameter gradients
        # optimizer.zero_grad(set_to_none=True)

    def log_scalar(self, x: torch.Tensor, name: str):
        if self.net.training:
            self.log_buffer[name].append(x.detach())

    def load(self, exp_dir: str, device: torch.device):
        path = os.path.join(exp_dir, "model.pt")
        state_dicts = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(state_dicts["model"])
