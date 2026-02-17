import torch

from collections import defaultdict
from omegaconf import DictConfig

from src.models.classifier import Classifier
from src.models.base_model import Model
from src.datasets import UnfoldingData
from src.networks import TransformerEncoder, LGATr
from src.utils.utils import load_model
from src.datasets.omnifold import OmniFoldObsProcess


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
        return isinstance(self.net, (TransformerEncoder, LGATr))

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

    def _forward_classifier(self, batch: UnfoldingData):

        if (batch.aux_x is not None) and (not self.classifier.lowlevel):

            # extract HLF features: mass, mult, width, tau21, ln_rho, zg
            # clone to avoid side effects on the original batch
            x = batch.aux_x[..., 1:].clone()

            # create a temporary object to hold features for the transform
            class FeatureContainer:
                def __init__(self, x):
                    self.x = x
                    self.z = torch.empty_like(x)

            features = FeatureContainer(x)

            # apply preprocessing
            # we move the transform to the device of the data
            transform = OmniFoldObsProcess().transforms[0].to(x.device)
            transform.forward(features)

            class BatchWrapper:
                def __init__(self, batch, x):
                    self.batch = batch
                    self.x = x

                def __getattr__(self, name):
                    return getattr(self.batch, name)

            return self.classifier(BatchWrapper(batch, features.x)), features.x
        return self.classifier(batch), None

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        # calculate gradnorm loss
        with torch.enable_grad():

            self.classifier.eval()  # disable batchnorm, dropout etc.

            # forward pass classifier
            lw_x, aux_x = self._forward_classifier(batch)
            lw_x = lw_x.squeeze(-1)

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
        return isinstance(self.net, (TransformerEncoder, LGATr))

    def forward(self, batch: UnfoldingData):
        """Return the part-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

        return self.net(batch.z)

    def _forward_classifier(self, batch: UnfoldingData):
        if hasattr(batch, "aux_x") and not self.classifier.lowlevel:

            # extract HLF features: mass, mult, width, tau21, ln_rho, zg
            # clone to avoid side effects on the original batch
            x = batch.aux_x[..., 1:].clone()

            # create a temporary object to hold features for the transform
            class FeatureContainer:
                def __init__(self, x):
                    self.x = x
                    self.z = torch.empty_like(x)

            features = FeatureContainer(x)

            # apply preprocessing
            # we move the transform to the device of the data
            transform = OmniFoldObsProcess().transforms[0].to(x.device)
            transform.forward(features)

            class BatchWrapper:
                def __init__(self, batch, x):
                    self.batch = batch
                    self.x = x

                def __getattr__(self, name):
                    return getattr(self.batch, name)

            return self.classifier(BatchWrapper(batch, features.x))
        return self.classifier(batch)

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        self.classifier.eval()  # disable batchnorm, dropout etc.
        with torch.no_grad():
            lw_x = self.classifier(batch).squeeze(-1).detach()

        # lw_x, aux_x = self._forward_classifier(batch)
        # lw_x = lw_x.squeeze(-1).detach()

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
        self.classifier.eval()
        with torch.no_grad():
            lw_x, aux_x = self._forward_classifier(batch)
            # lw_x = lw_x.squeeze(-1)
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
        if self.lowlevel:
            assert (
                aux_x is not None
            ), "Auxiliary features required for lowlevel RKHSUnfolder"
            K = self.rbf_kernel_matrix(
                aux_x[1:], None, lengthscale=self.cfg.scale
            )  # (N,N)
        else:
            K = self.rbf_kernel_matrix(
                batch.x,
                None,
                lengthscale=self.cfg.scale,
            )  # (N,N)
        K.diagonal().zero_()

        # empirical RKHS quadratic loss: (1/N^2) r^T K r
        # r^T K r = sum_{i,j} r_i K_ij r_j
        # implement as (r.T @ K @ r) scalar
        B = len(batch)
        # loss_quad = (r.transpose(-2, -1) @ K @ r).squeeze().abs().sqrt() / (B * (B - 1))
        loss_quad = (loss_reg.transpose(-2, -1) @ K @ loss_reg).squeeze() / (
            B * (B - 1)
        )

        loss_quad = loss_quad.clamp(min=0.0)

        if self.net.ensembled:
            loss_quad = loss_quad.sum(0)  # sum over ensemble members

        # log the inverse regression loss
        # with torch.no_grad():
        #     loss_mlc = (-lw_z.exp() * lw_x - (1 - lw_x.exp())) / 2
        # self.log_scalar(loss_mlc, "loss_mlc")
        # with torch.no_grad():
        #     loss_omnifold = (-lw_x.exp() * lw_z - (1 - lw_z.exp())) / 2
        # self.log_scalar(loss_omnifold, "loss_omnifold")

        self.log_scalar(loss_reg.mean(), "r2_mean")
        self.log_scalar(K.sum() / (B * (B - 1)), "kernel_mean")

        return loss_quad


# class SmearUnfolder(Unfolder):

#     def __init__(self, cfg: DictConfig):
#         super().__init__(cfg)
#         self.kernel = torch.distributions.Normal(loc=0.0, scale=cfg.kernel_scale)

#     def batch_loss(self, batch: UnfoldingData):

#         # restrict to simulation only
#         batch = batch[batch.labels == 0]

#         with torch.no_grad():
#             # forward pass classifier
#             self.classifier.eval()
#             C = self.classifier(batch).squeeze(-1)
#             if self.classifier.ensembled:
#                 C = C.mean(0)

#             # smearing kernel
#             K = self.kernel.log_prob(torch.cdist(batch.x, batch.x)).exp()
#             # K.diagonal().zero_()  # zero out self-pairing

#         # forward pass unfolder
#         lw_z = self.forward(batch).squeeze(-1)
#         print(f"{lw_z.shape, lw_z=}")

#         # smear
#         # w_x = (K @ lw_z.exp()) / K.sum(dim=1)
#         lw_x = (K @ lw_z.exp()).log() - K.sum(dim=1).log()

#         print(f"{K.shape, K=}")
#         print(f"{lw_x.shape, lw_x=}")

#         # calculate regression loss
#         batch_dim = (0, 1) if self.ensembled else 0
#         match self.cfg.loss:

#             case "mse":
#                 # loss_reg = (lw_x.exp() - C.exp()).pow(2).mean(batch_dim)
#                 loss_reg = (lw_x - C).pow(2).mean(batch_dim)
#                 print(f"{loss_reg=}")
#             case "bce":
#                 loss_reg = (
#                     (C.exp() + 1) * torch.nn.functional.softplus(-lw_x) + lw_x
#                 ).mean(batch_dim) / 2
#                 # loss_reg = - C.exp() * w_x.log().sigmoid()
#             case "mlc":
#                 loss_reg = (-C.exp() * lw_x - (1 - lw_x.exp())).mean(batch_dim) / 2

#         self.log_scalar(loss_reg, "loss_reg")

#         # return None

#         return loss_reg


# class RKHSUnfolder2(Unfolder):

#     def rbf_kernel_matrix(self, X, Y=None, lengthscale=1.0, eps=1e-8):
#         """
#         Compute RBF (Gaussian) kernel matrix between X and Y:
#         K_ij = exp(-0.5 * ||x_i - y_j||^2 / lengthscale^2)
#         X: (N, d)
#         Y: (M, d) or None -> Y = X
#         Returns: (N, M)
#         """
#         if Y is None:
#             Y = X
#         # squared norms
#         X_sq = (X * X).sum(dim=-1, keepdim=True)  # (N,1)
#         Y_sq = (Y * Y).sum(dim=-1, keepdim=True)  # (M,1)
#         # pairwise squared distance: (N, M)
#         dist2 = X_sq - 2.0 * (X @ Y.transpose(-2, -1)) + Y_sq.transpose(-2, -1)

#         # numerical stability: clamp small negatives to zero
#         dist2 = torch.clamp(dist2, min=0.0)
#         K = torch.exp(-0.5 * dist2 / (lengthscale**2 + eps))  # gaussian kernel
#         # C = 2 * X.size(-1) * lengthscale**2 # inverse multiquadratic kernel
#         # K = C / (dist2 + C)  # inverse multiquadratic kernel

#         return K

#     def batch_loss(self, batch: UnfoldingData):

#         # restrict to simulation only
#         batch = batch[batch.labels == 0]

#         # forward pass classifier
#         self.classifier.eval()
#         with torch.no_grad():
#             lw_x = self.classifier(batch)

#         if self.classifier.ensembled:
#             lw_x = lw_x.mean(0)

#         # forward pass unfolder
#         lw_z = self.forward(batch)

#         # compute pointwise reg loss
#         match self.cfg.loss:
#             case "mse":
#                 loss_reg = lw_z.exp() - lw_x.exp()
#             case "bce":
#                 loss_reg = (1 - (lw_z - lw_x).exp()) / (1 + lw_x.exp())
#             case "mlc":
#                 loss_reg = 1 - (lw_z - lw_x).exp()

#         # loss_reg = loss_reg.tanh()

#         # r = 1 - (lw_z - lw_x).exp()

#         # compute kernel matrix K (N,N)
#         if self.lowlevel:
#             K = self.rbf_kernel_matrix(
#                 batch.cond_x, None, lengthscale=self.cfg.scale
#             )  # (N,N)
#         else:
#             K = self.rbf_kernel_matrix(
#                 batch.x,
#                 None,
#                 lengthscale=self.cfg.scale,
#             )  # (N,N)
#         K.diagonal().zero_()

#         # empirical RKHS quadratic loss: (1/N^2) r^T K r
#         # r^T K r = sum_{i,j} r_i K_ij r_j
#         # implement as (r.T @ K @ r) scalar
#         B = len(batch)
#         # loss_quad = (r.transpose(-2, -1) @ K @ r).squeeze().abs().sqrt() / (B * (B - 1))
#         loss_quad = (loss_reg.transpose(-2, -1) @ K @ loss_reg).squeeze() / (
#             B * (B - 1)
#         )

#         loss_quad = loss_quad.clamp(min=0.0)

#         if self.net.ensembled:
#             loss_quad = loss_quad.sum(0)  # sum over ensemble members

#         self.log_scalar(loss_reg.mean(), "r2_mean")
#         self.log_scalar(K.sum() / (B * (B - 1)), "kernel_mean")

#         return loss_quad


# # class PsiGradnormUnfolder(Unfolder):

# #     def __init__(self, cfg: DictConfig):

# #         super().__init__(cfg)

# #         assert self.classifier.ensembled, "Classifier must be ensembled for PsiGradnormUnfolder"

# #         # Initialize psi
# #         # Assuming self.classifier.ensembled is the integer number of members

# #         # Initialize psi
# #         self.psi = torch.nn.Parameter(
# #             torch.load(
# #                 os.path.join(self.cfg.psi_path, "model.pt"),
# #                 weights_only=False,
# #                 map_location="cpu"
# #             )['model']['psi'],
# #             requires_grad=True
# #         )
# #         # n_members = self.classifier.ensembled
# #         # self.psi = torch.nn.Parameter(torch.ones(n_members) / n_members, requires_grad=True)

# #     # @property
# #     # def trainable_parameters(self):
# #     #     # Determine which parameters are trainable (unfolder net + psi)
# #     #     import itertools
# #     #     return itertools.chain(
# #     #         (p for p in self.net.parameters() if p.requires_grad),
# #     #         [self.psi]
# #     #     )

# #     def batch_loss(self, batch: UnfoldingData):

# #         # restrict to simulation only
# #         batch = batch[batch.labels == 0]

# #         # forward pass unfolder
# #         lw_z = self.forward(batch).squeeze(-1)

# #         # calculate gradnorm loss
# #         with torch.enable_grad():

# #             self.classifier.eval()  # disable batchnorm, dropout etc.

# #             # forward pass classifier
# #             # lw_x shape: [E, B] (assuming E members) or [B] if not ensembled (but we asserted it is)
# #             with torch.no_grad():
# #                 lw_x = self.classifier(batch).squeeze(-1)

# #             # Dot psi with lw_x
# #             # psi: [E], lw_x: [E, B] -> [B]
# #             # using einsum for clarity: 'e, eb -> b'
# #             lw_x_combined = torch.einsum('e, eb -> b', self.psi, lw_x)
# #             # lw_x_combined = torch.einsum('e, eb -> b', self.psi.softmax(0), lw_x)

# #             # calculate regression loss
# #             match self.cfg.loss:

# #                 case "mse":
# #                     loss_reg = (lw_z.exp() - lw_x_combined.exp()).pow(2)
# #                 case "bce":
# #                     loss_reg = (
# #                         (lw_z.exp() + 1) * torch.nn.functional.softplus(-lw_x_combined) + lw_x_combined
# #                     ) / 2
# #                 case "mlc":
# #                     loss_reg = (-lw_z.exp() * lw_x_combined - (1 - lw_x_combined.exp())) / 2


# #             loss_reg = loss_reg.mean()

# #             # take gradients w.r.t psi
# #             grads_psi = torch.autograd.grad(
# #                 loss_reg,
# #                 self.psi,
# #                 create_graph=True,
# #             )
# #             # grads_psi will be a tuple containing one tensor of shape [E]

# #         # sum gradient norms
# #         match self.cfg.norm:
# #             case "l1":
# #                 norm = lambda g: g.abs()
# #             case "l2":
# #                 norm = lambda g: g.pow(2)

# #         loss_gradnorm = norm(grads_psi[0]).sum()

# #         self.log_scalar(loss_reg, "loss_reg")
# #         self.log_scalar(loss_gradnorm, "loss_gradnorm")
# #         self.log_scalar(self.psi.abs().mean(), "psi")

# #         # scale result
# #         loss_gradnorm = loss_gradnorm * 1e3

# #         if self.ensembled:
# #              loss_gradnorm = loss_gradnorm * self.ensembled

# #         return loss_gradnorm

# from torch.func import vjp, functional_call


class JVPUnfolder(Unfolder):

    def batch_loss(self, batch: UnfoldingData):

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        # calculate gradnorm loss
        self.classifier.eval()
        params = dict(self.classifier.named_parameters())
        with torch.enable_grad():
            lw_x, vjp_fn = vjp(
                lambda p: functional_call(self.classifier, p, batch).squeeze(-1),
                params,
            )
            if self.classifier.ensembled:

                if self.cfg.joint_ensembling:
                    # combine classifier and unfolder ensemble members elementwise
                    assert self.ensembled == self.classifier.ensembled
                else:
                    # average over classifier ensemble
                    lw_x = lw_x.mean(0)

        residual = 1 - (lw_z - lw_x.detach()).exp()
        grads_x = vjp_fn(residual)[0]

        # sum gradient norms
        match self.cfg.norm:
            case "l1":
                norm = lambda g: g.abs()
            case "l2":
                norm = lambda g: g.pow(2)

        loss_gradnorm = sum(norm(g).sum() for g in grads_x.values()) / (
            len(batch) * self.num_params_cls
        )

        # self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")

        # scale result
        loss_gradnorm = loss_gradnorm * 1e3
        if self.ensembled:
            # make lr scale independent of ensemble size
            loss_gradnorm = loss_gradnorm * self.ensembled

        return loss_gradnorm

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if (p.requires_grad and p.numel() > 0))


class NTKUnfolder(Unfolder):

    def batch_loss(self, batch: UnfoldingData):

        assert not self.ensembled

        # restrict to simulation only
        batch = batch[batch.labels == 0]

        # forward pass unfolder
        lw_z = self.forward(batch).squeeze(-1)

        # prepare classifier inputs
        self.classifier.eval()
        params = dict(self.classifier.named_parameters())
        buffers = dict(self.classifier.named_buffers())

        scale = 1e6 / (self.num_params_cls * len(batch))
        # scale = 1 / (self.num_params_cls * len(batch))
        loss_gradnorm = JacobianReg.apply(
            lw_z, batch, self.classifier, params, buffers, scale
        )

        # self.log_scalar(loss_reg, "loss_reg")
        self.log_scalar(loss_gradnorm, "loss_gradnorm")

        # # scale result
        # loss_gradnorm = loss_gradnorm * 1e3
        # if self.ensembled:
        #     # make lr scale independent of ensemble size
        #     loss_gradnorm = loss_gradnorm * self.ensembled

        return loss_gradnorm

    @property
    def trainable_parameters(self):
        return (p for p in self.net.parameters() if (p.requires_grad and p.numel() > 0))


from torch.func import functional_call, vjp, jvp


# class JacobianReg(torch.autograd.Function):
#     """
#     TODO: Document
#     """

#     @staticmethod
#     def forward(ctx, lw_z, batch, classifier, params, buffers, scale=1):

#         # define functional wrapper for classifier
#         def call_classifier(p, b):
#             lw_x = functional_call(classifier, (p, b), batch).squeeze(-1)
#             if classifier.ensembled:
#                 lw_x = lw_x.mean(0)
#             return lw_x

#         # vjp pass (computes forward + gradient prep)
#         with torch.no_grad():
#             # call the classifier
#             lw_x, vjp_fn = vjp(lambda p: call_classifier(p, buffers), params)

#             # compute residual (functional derivative of regression loss)
#             R = 1 - (lw_z - lw_x.detach()).exp()

#             # project onto neural tangent space
#             proj_grads = vjp_fn(R * scale)[0]

#         # compute norm
#         # loss = sum(g.pow(2).sum() for g in proj_grads.values() if g is not None) # L2
#         loss = sum(g.abs().sum() for g in proj_grads.values() if g is not None)  # L1

#         # save state for backward
#         ctx.keys = list(params.keys())
#         ctx.classifier = classifier
#         ctx.buffers = buffers
#         ctx.batch = batch
#         # save batch and the projected gradients
#         ctx.save_for_backward(*[proj_grads[k] for k in ctx.keys])

#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):

#         # unpack tensors from context
#         grad_tensors = ctx.saved_tensors

#         # extract tangents and parameters
#         proj_grads = dict(zip(ctx.keys, grad_tensors))
#         params = dict(ctx.classifier.named_parameters())

#         # compute d(Loss)/dR = 2 * J * (J^T * R) with jvp (forward mode AD)
#         def call_classifier(p):
#             lw_x = functional_call(ctx.classifier, (p, ctx.buffers), ctx.batch).squeeze(
#                 -1
#             )
#             if ctx.classifier.ensembled:
#                 lw_x = lw_x.mean(0)
#             return lw_x

#         _, jvp_out = jvp(call_classifier, (params,), (proj_grads,))
#         grad_R = 2 * jvp_out * grad_output

#         # gradients for (lw_z, batch, classifier, params, buffers, scale)
#         # only return gradient for z_out (which flows back to unfolder).
#         return grad_R, None, None, None, None, None


class JacobianReg(torch.autograd.Function):
    """
    Jacobian Regularization with optimized Backprop-through-Backprop.
    Minimizes L = || scale * J^T * R ||^2
    """

    @staticmethod
    def forward(ctx, lw_z, batch, classifier, params, buffers, scale=1):

        # 1. Define functional wrapper for classifier
        def call_classifier(p, b):
            lw_x = functional_call(classifier, (p, b), batch).squeeze(-1)
            if classifier.ensembled:
                lw_x = lw_x.mean(0)
            return lw_x

        # 2. VJP Pass (computes forward + gradient prep)
        with torch.no_grad():
            # Run classifier and prepare gradient function
            # lw_x inherits shape from batch (e.g., [B2])
            lw_x, vjp_fn = vjp(lambda p: call_classifier(p, buffers), params)

            # Compute residual R.
            # lw_z might be larger [B1, B2], so this subtraction broadcasts.
            R = 1 - (lw_z - lw_x.detach()).exp()

            # --- SHAPE FIX: Sum over broadcasted dimensions ---
            # If lw_z is [B1, B2] and lw_x is [B2], we must sum R over dim 0
            # before passing to vjp, or PyTorch will error.
            if R.shape != lw_x.shape:
                R_vjp = R.sum(dim=0)
            else:
                R_vjp = R

            # Project onto neural tangent space (J^T * R)
            # We apply scale here. Tangents = scale * J^T * R
            proj_grads = vjp_fn(R_vjp * scale)[0]

        # 3. Compute Norm
        # We use L2 norm because the backward pass (2 * J * v) assumes x^2 derivatives.
        loss = sum(g.pow(2).sum() for g in proj_grads.values() if g is not None)
        # loss = sum(g.abs().sum() for g in proj_grads.values() if g is not None) # L1 (requires changing backward)

        # 4. Save state for backward
        ctx.keys = list(params.keys())
        ctx.classifier = classifier
        ctx.buffers = buffers
        ctx.batch = batch
        ctx.scale = scale

        # We must save lw_z and lw_x to compute dR/dlw_z in backward
        ctx.save_for_backward(lw_z, lw_x, *[proj_grads[k] for k in ctx.keys])

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # 1. Unpack tensors
        lw_z = ctx.saved_tensors[0]
        lw_x = ctx.saved_tensors[1]
        grad_tensors = ctx.saved_tensors[2:]

        # Extract tangents and parameters
        proj_grads = dict(zip(ctx.keys, grad_tensors))
        params = dict(ctx.classifier.named_parameters())

        # 2. JVP Pass (Forward Mode AD)
        # Compute J * proj_grads
        def call_classifier(p):
            lw_x = functional_call(ctx.classifier, (p, ctx.buffers), ctx.batch).squeeze(
                -1
            )
            if ctx.classifier.ensembled:
                lw_x = lw_x.mean(0)
            return lw_x

        _, jvp_out = jvp(call_classifier, (params,), (proj_grads,))

        # 3. Compute Gradient Chain
        # L = || v ||^2 where v = scale * J^T * R
        # dL/dv = 2 * v
        # dv/dR = scale * J
        # dL/dR = (dL/dv)(dv/dR) = 2 * v * scale * J
        # Since jvp_out = J * v, we have:
        # dL/dR = 2 * scale * jvp_out
        grad_R = 2 * ctx.scale * jvp_out * grad_output

        # 4. Compute dL/dlw_z
        # dR/dlw_z = -exp(lw_z - lw_x)
        # This subtraction broadcasts correctly if lw_z is [B1, B2] and lw_x is [B2]
        dR_dlw_z = -(lw_z - lw_x).exp()

        # Final chain rule
        # grad_R (shape of lw_x) * dR_dlw_z (shape of lw_z) -> broadcasts correctly
        grad_lw_z = grad_R * dR_dlw_z

        # Return gradients for (lw_z, batch, classifier, params, buffers, scale)
        return grad_lw_z, None, None, None, None, None
