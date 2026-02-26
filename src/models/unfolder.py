import torch

from collections import defaultdict
from omegaconf import DictConfig

from src.models.classifier import Classifier
from src.models.base_model import Model
from src.datasets import UnfoldingData
from src.networks import TransformerEncoder, LGATr
from src.utils.utils import load_model


class Unfolder(Model):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        self.cfg = cfg

        # load pretrained classifier
        self.classifier, self.cfg_classifier = load_model(
            self.cfg.cls_path, Classifier, freeze=False
        )
        # self.params_cls = dict(self.classifier.named_parameters())
        self.params_cls = list(self.classifier.parameters())
        self.num_params_cls = sum(p.numel() for p in self.params_cls)

        # logging
        self.log_buffer = defaultdict(list)

        # ensembling
        self.ensembled = self.net.ensembled

    @property
    def lowlevel(self):
        return self.net.lowlevel

    def forward(self, batch: UnfoldingData):
        """Return the part-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

        if batch.aux_z is not None:

            # create a temporary object to hold features for the transform
            class FeatureContainer:
                def __init__(self, z):
                    self.z = z
                    self.x = torch.empty_like(z)

            features = FeatureContainer(batch.aux_z[..., 1:].clone())

            # apply preprocessing
            # we move the transform to the device of the data
            transform = OmniFoldObsProcess().transforms[0].to(batch.device)
            transform.forward(features)

            return self.net(features.z)

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

            if self.classifier.ensembled:

                if self.cfg.joint_ensembling:
                    # combine classifier and unfolder ensemble members elementwise
                    assert self.ensembled == self.classifier.ensembled
                else:
                    # average over classifier ensemble
                    lw_x = lw_x.mean(0)

            if self.cfg.norm_target:
                norm = lw_x.clone().detach().exp().mean()
                lw_x -= norm.log()  # ensure norm is precisely 1 batchwise

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

            # sample weights
            if batch.sample_logweights is not None:
                loss_reg = loss_reg * batch.sample_logweights.exp()

            # average
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
                allow_unused=True,
            )

        # sum gradient norms
        match self.cfg.norm:
            case "l1":
                norm = lambda g: g.abs()
            case "l2":
                norm = lambda g: g.pow(2)
            case "l12":
                norm = lambda g: g.pow(2)

        loss_gradnorm = (
            sum(norm(g).sum() for g in grads_x if g is not None) / self.num_params_cls
        )

        if self.cfg.norm == "l12":
            loss_gradnorm = loss_gradnorm.sqrt()

        self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")
        
        # # log the omnifold loss
        # with torch.no_grad():
        #     loss_omnifold = (-lw_x.exp() * lw_z - (1 - lw_z.exp())) / 2
        # self.log_scalar(loss_omnifold, "loss_omnifold")

        # scale result
        loss_gradnorm = loss_gradnorm * 1e3
        if self.ensembled:
            # make lr scale independent of ensemble size
            loss_gradnorm = loss_gradnorm * self.ensembled

        return loss_gradnorm

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if (p.requires_grad and p.numel() > 0))


class OmniFolder(Model):

    def __init__(self, cfg: DictConfig):

        super().__init__(cfg)

        self.cfg = cfg

        # load pretrained classifier
        self.classifier, self.cfg_classifier = load_model(
            self.cfg.cls_path, Classifier, freeze=True
        )

        # logging
        self.log_buffer = defaultdict(list)

        # ensembling
        self.ensembled = self.net.ensembled

    @property
    def lowlevel(self):
        return self.net.lowlevel

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
        with torch.no_grad():
            lw_x = self.classifier(batch).squeeze(-1)


        if self.classifier.ensembled:

            if self.cfg.joint_ensembling:
                # combine classifier and unfolder ensemble members elementwise
                assert self.ensembled == self.classifier.ensembled
            else:
                # average over classifier ensemble
                lw_x = lw_x.mean(0)

        # calculate regression loss
        match self.cfg.loss:

            case "mse":
                loss_reg = (lw_z - lw_x).pow(2)
            case "mse2":
                loss_reg = (lw_z.exp() - lw_x.exp()).pow(2)
            case "bce":
                loss_reg = (
                    (lw_x.exp() + 1) * torch.nn.functional.softplus(-lw_z) + lw_z
                ) / 2
            case "mlc":
                loss_reg = (-lw_x.exp() * lw_z - (1 - lw_z.exp())) / 2

        # sample weights
        if batch.sample_logweights is not None:
            loss_reg = loss_reg * batch.sample_logweights.exp()

        # average
        batch_dim = (0, 1) if self.ensembled else 0
        loss_reg = loss_reg.mean(batch_dim)
        self.log_scalar(loss_reg, "loss_reg")

        # # log the inverse regression loss
        # with torch.no_grad():
        #     loss_mlc = (-lw_z.exp() * lw_x - (1 - lw_x.exp())) / 2
        # self.log_scalar(loss_mlc, "loss_mlc")

        return loss_reg

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if p.requires_grad)


class RKHSUnfolder(Unfolder):

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass classifier
        self.classifier.eval()
        with torch.no_grad():
            lw_x = self.classifier(batch).squeeze(-1)
            if self.classifier.ensembled:

                if self.cfg.joint_ensembling:
                    # combine classifier and unfolder ensemble members elementwise
                    assert self.ensembled == self.classifier.ensembled
                else:
                    # average over classifier ensemble
                    lw_x = lw_x.mean(0)

        # forward pass unfolder
        lw_z = self.forward(batch)  # .squeeze(-1)

        # compute pointwise reg loss
        match self.cfg.loss:
            case "mse":
                loss_reg = lw_z.exp() - lw_x.exp()
            case "bce":
                loss_reg = (1 - (lw_z - lw_x).exp()) / (1 + lw_x.exp())
            case "mlc":
                loss_reg = 1 - (lw_z - lw_x).exp()

        # sample weights
        if batch.sample_logweights is not None:
            loss_reg = loss_reg * batch.sample_logweights.exp().unsqueeze(-1)

        # compute kernel matrix K (N,N)
        # assert not self.lowlevel
        dist = torch.cdist(batch.x, batch.x, p=self.cfg.kernel_p)  # (N, N)
        K = torch.exp(-0.5 * dist.pow(2) / self.cfg.kernel_scale**2)
        K.diagonal().zero_()

        B = len(batch)
        norm2 = (loss_reg.transpose(-2, -1) @ K @ loss_reg).squeeze(-1) / (B * (B - 1))

        norm = norm2.clamp(0)  # numerical stability

        if self.net.ensembled:
            norm = norm.sum(0)  # sum over ensemble members

        # log the alternative losses
        with torch.no_grad():
            loss_mlc = (-lw_z.exp() * lw_x - (1 - lw_x.exp())) / 2
            loss_omnifold = (-lw_x.exp() * lw_z - (1 - lw_z.exp())) / 2
        self.log_scalar(loss_mlc.mean(), "loss_mlc")
        self.log_scalar(loss_omnifold.mean(), "loss_omnifold")

        self.log_scalar(loss_reg.mean(), "r2_mean")

        return norm