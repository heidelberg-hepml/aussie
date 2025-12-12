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
        self.num_params_cls = sum(p.numel() for p in self.params_cls.values())

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
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

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
            lw_x = self.classifier(batch).squeeze(-1)

            # average over classifier ensemble
            if self.classifier.ensembled:
                lw_x = lw_x.mean(0)

            # calculate regression loss
            match self.cfg.loss:

                case "mse":
                    loss_reg = (lw_z.exp() - lw_x.exp()).pow(2)
                case "bce":
                    loss_reg = (
                        (lw_z.exp() + 1) * torch.nn.functional.softplus(-lw_x) + lw_x
                    ) / 2
                case "mlc":
                    loss_reg = (-lw_z.exp() * lw_x - (1 - lw_x.exp())) / 2

            batch_dim = int(bool(self.ensembled))
            loss_reg = loss_reg.mean(batch_dim)

            # take gradients (in parallel if ensembled)
            grad_outputs = (
                torch.eye(self.ensembled, device=lw_z.device, dtype=lw_z.dtype)
                if self.ensembled
                else None
            )
            grads_x = torch.autograd.grad(
                loss_reg,
                self.classifier.trainable_parameters,
                create_graph=True,
                grad_outputs=grad_outputs,
                is_grads_batched=self.ensembled,
            )

        # sum gradient norms
        match self.cfg.norm:
            case "l1":
                norm = lambda g: g.abs()
            case "l2":
                norm = lambda g: g.pow(2)

        loss_gradnorm = sum(norm(g).sum() for g in grads_x) / self.num_params_cls

        self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")

        # scale result
        loss_gradnorm = loss_gradnorm * 1e3
        if self.ensembled:
            # make lr scale independent of ensemble size
            loss_gradnorm = loss_gradnorm * self.ensembled

        return loss_gradnorm

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
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

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


class RKHSUnfolder(Unfolder):

    def init__(self, cfg: DictConfig):
        super().init(cfg)
        self.kernel = torch.distributions.Normal(loc=0.0, scale=cfg.kernel_scale)

    # def rbf_kernel_matrix(self, X, Y=None, lengthscale=1.0):
    #     # X: (N, d), Y: (M, d) or None
    #     if Y is None:
    #         Y = X
    #     X_sq = (X**2).sum(dim=1, keepdim=True)  # (N,1)
    #     Y_sq = (Y**2).sum(dim=1, keepdim=True)  # (M,1)
    #     # pairwise squared distances: (N, M)
    #     dist2 = X_sq - 2.0 * X @ Y.t() + Y_sq.t()
    #     K = torch.exp(-0.5 * dist2 / (lengthscale**2))
    #     return K

    def rbf_kernel_matrix(self, X, Y=None, lengthscale=1.0, eps=1e-8):
        """
        Compute RBF (Gaussian) kernel matrix between X and Y:
        K_ij = exp(-0.5 * ||x_i - y_j||^2 / lengthscale^2)
        X: (N, d)
        Y: (M, d) or None -> Y = X
        Returns: (N, M)
        """
        if Y is None:
            Y = X
        # squared norms
        X_sq = (X * X).sum(dim=-1, keepdim=True)  # (N,1)
        Y_sq = (Y * Y).sum(dim=-1, keepdim=True)  # (M,1)
        # pairwise squared distance: (N, M)
        dist2 = X_sq - 2.0 * (X @ Y.transpose(-2, -1)) + Y_sq.transpose(-2, -1)

        # numerical stability: clamp small negatives to zero
        dist2 = torch.clamp(dist2, min=0.0)
        K = torch.exp(-0.5 * dist2 / (lengthscale**2 + eps))  # gaussian kernel
        # C = 2 * X.size(-1) * lengthscale**2 # inverse multiquadratic kernel
        # K = C / (dist2 + C)  # inverse multiquadratic kernel

        return K

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass classifier
        with torch.no_grad():
            lw_x = self.classifier(batch)

        if self.classifier.ensembled:
            lw_x = lw_x.mean(0)

        # forward pass unfolder
        lw_z = self.forward(batch)

        # compute residuals
        r = 1 - (lw_z - lw_x).exp()

        # compute kernel matrix K (N,N)
        if self.lowlevel:
            K = self.rbf_kernel_matrix(
                batch.cond_x, None, lengthscale=self.cfg.scale
            )  # (N,N)
        else:
            K = self.rbf_kernel_matrix(
                batch.x, None, lengthscale=self.cfg.scale
                None,
                lengthscale=self.cfg.scale,
            )  # (N,N)
        K.diagonal().zero_()

        # empirical RKHS quadratic loss: (1/N^2) r^T K r
        # r^T K r = sum_{i,j} r_i K_ij r_j
        # implement as (r.T @ K @ r) scalar
        B = len(batch)
        # loss_quad = (r.transpose(-2, -1) @ K @ r).squeeze().abs().sqrt() / (B * (B - 1))
        loss_quad = (r.transpose(-2, -1) @ K @ r).squeeze() / (B * (B - 1))

        if self.net.ensembled:
            loss_quad = loss_quad.sum(0)  # sum over ensemble members

        self.log_scalar(r.mean(), "r2_mean")
        self.log_scalar(K.sum() / (B * (B - 1)), "kernel_mean")

        return loss_quad
