import os
import torch

from collections import defaultdict
from contextlib import nullcontext
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.func import functional_call

from src.models.classifier import Classifier
from src.models.base_model import Model
from src.datasets import UnfoldingData
from src.utils.utils import load_model

# log = logging.getLogger("Model")


class Unfolder(Model):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        self.cfg = cfg

        # load pretrained classifier
        self.classifier, self.cfg_classifier = load_model(
            self.cfg.cls_path, Classifier, freeze=False
        )
        self.params_cls = dict(self.classifier.named_parameters())
        # for p in self.classifier.parameters():
        #     p.requires_grad = True

        # logging
        self.log_buffer = defaultdict(list)

        # regression optimization steps

        # bayesian
        self.bayesian = self.net.bayesian
        if self.bayesian:
            self.register_buffer("train_size", torch.zeros(()))

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        lw_z = self.net(batch.z).squeeze(-1)

        with torch.enable_grad():
            self.classifier.eval()
            lw_x = functional_call(self.classifier, self.params_cls, (batch,)).squeeze(
                -1
            )

            match self.cfg.loss:

                case "mse":
                    loss_reg = (lw_z - lw_x).pow(2).mean(0)
                case "bce":
                    loss_reg = (
                        (lw_z.exp() + 1) * torch.nn.functional.softplus(-lw_x) + lw_x
                    ).mean(0) / 2
                case "mlc":
                    loss_reg = (-lw_z.exp() * lw_x - (1 - lw_x.exp())).mean(0) / 2

            grads_x = torch.autograd.grad(
                loss_reg, self.params_cls.values(), create_graph=True
            )

        loss_gradnorm = sum(g.pow(2).sum() for g in grads_x)

        self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")

        return loss_gradnorm

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)
