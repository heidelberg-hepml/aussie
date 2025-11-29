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
from src.networks import TransformerEncoder
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

        # logging
        self.log_buffer = defaultdict(list)

        # bayesian
        self.bayesian = self.net.bayesian
        if self.bayesian:
            self.register_buffer("train_size", torch.zeros(()))

        # ensembling
        self.ensembled = self.net.ensembled

    @property
    def lowlevel(self):
        return isinstance(self.net, TransformerEncoder)

    def forward(self, batch: UnfoldingData):
        """Return the part-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, mask=batch.z[..., 0] != 0)

        return self.net(batch.z)

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        # calculate gradnorm loss
        with torch.enable_grad():

            self.classifier.eval()  # disable batchnorm, dropout etc.

            # forward pass classifier
            lw_x = functional_call(self.classifier, self.params_cls, (batch,)).squeeze(
                -1
            )

            # average over classifier ensemble
            if self.classifier.ensembled:
                lw_x = lw_x.mean(0)

            # calculate regression loss
            batch_dim = int(bool(self.ensembled))
            match self.cfg.loss:

                case "mse":
                    loss_reg = (lw_z - lw_x).pow(2).mean(batch_dim)
                case "mse2":
                    loss_reg = (lw_z.exp() - lw_x.exp()).pow(2).mean(batch_dim)
                case "bce":
                    loss_reg = (
                        (lw_z.exp() + 1) * torch.nn.functional.softplus(-lw_x) + lw_x
                    ).mean(batch_dim) / 2
                case "mlc":
                    loss_reg = (-lw_z.exp() * lw_x - (1 - lw_x.exp())).mean(
                        batch_dim
                    ) / 2

            # take gradients (in parallel if ensembled)
            grad_outputs = (
                torch.eye(self.ensembled, device=lw_z.device, dtype=lw_z.dtype)
                if self.ensembled
                else None
            )
            grads_x = torch.autograd.grad(
                loss_reg,
                self.params_cls.values(),
                create_graph=True,
                grad_outputs=grad_outputs,
                is_grads_batched=self.ensembled,
            )

        # sum gradient norms
        match self.cfg.norm:

            case "l1":
                loss_gradnorm = sum(g.abs().sum() for g in grads_x)
            case "l2":
                loss_gradnorm = sum(g.pow(2).sum() for g in grads_x)

        # normalize result
        num_params = sum(w.numel() for w in self.params_cls.values())
        loss_gradnorm = loss_gradnorm / num_params  # don't divide by self.ensembled

        self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")

        return loss_gradnorm * 1e6

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)


class OmniFolder(Model):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        self.cfg = cfg

        # load pretrained classifier
        self.classifier, self.cfg_classifier = load_model(
            self.cfg.cls_path, Classifier, freeze=False
        )

        # logging
        self.log_buffer = defaultdict(list)

        # bayesian
        self.bayesian = self.net.bayesian
        if self.bayesian:
            self.register_buffer("train_size", torch.zeros(()))

        # ensembling
        self.ensembled = self.net.ensembled

    @property
    def lowlevel(self):
        return isinstance(self.net, TransformerEncoder)

    def forward(self, batch: UnfoldingData):
        """Return the part-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, mask=batch.z[..., 0] != 0)

        return self.net(batch.z)

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        self.classifier.eval()  # disable batchnorm, dropout etc.
        lw_x = self.classifier(batch).squeeze(-1).detach()

        # average over classifier ensemble
        if self.classifier.ensembled:
            lw_x = lw_x.mean(0)

        # calculate regression loss
        batch_dim = (0, 1) if self.ensembled else 0
        match self.cfg.loss:

            case "mse":
                loss_reg = (lw_z - lw_x).pow(2).mean(batch_dim)
            case "mse2":
                loss_reg = (lw_z.exp() - lw_x.exp()).pow(2).mean(batch_dim)
            case "bce":
                loss_reg = (
                    (lw_x.exp() + 1) * torch.nn.functional.softplus(-lw_z) + lw_z
                ).mean(batch_dim) / 2
            case "mlc":
                loss_reg = (-lw_x.exp() * lw_z - (1 - lw_z.exp())).mean(batch_dim) / 2

        self.log_scalar(loss_reg, "loss_reg")

        return loss_reg

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)
