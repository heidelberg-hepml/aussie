import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.networks import TransformerEncoder, LGATr
from src.datasets import UnfoldingData


class Classifier(Model):

    @property
    def lowlevel(self):
        return isinstance(
            self.net, (TransformerEncoder, LGATr)
        )  # TODO: net as attribute in network

    def batch_loss(self, batch: UnfoldingData) -> torch.Tensor:

        # forward pass
        logits = self.forward(batch).squeeze(-1)

        # split by labels to handle imbalanced classes
        mask_sim = batch.labels == 0
        logits_sim = logits[..., mask_sim]  # ellipsis to handle ensembling
        logits_dat = logits[..., ~mask_sim]

        match self.cfg.loss:

            case "bce":
                loss_dat = F.binary_cross_entropy_with_logits(
                    logits_dat, torch.ones_like(logits_dat), reduction="none"
                )
                loss_sim = F.binary_cross_entropy_with_logits(
                    logits_sim, torch.zeros_like(logits_sim), reduction="none"
                )
            case "mlc":
                loss_dat = -logits_dat + (-logits_dat).exp()
                loss_sim = logits_sim + logits_sim.exp()
            case _:
                raise ValueError

        # sample weights
        if batch.sample_logweights is not None:
            sample_weights = batch.sample_logweights.exp()
            loss_dat = loss_dat * sample_weights[~mask_sim]
            loss_sim = loss_sim * sample_weights[mask_sim]

        # average loss
        mean_dims = (0, 1) if self.ensembled else 0
        loss = (loss_dat.mean(mean_dims) + loss_sim.mean(mean_dims)) / 2

        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        return loss

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.x, c=batch.cond_x, mask=batch.mask_x)

        return self.net(batch.x)

    def weight(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the event-level data-to-sim likelihood ratio"""
        logits = self.forward(batch)
        return logits.exp()

    def prob(self, batch: UnfoldingData) -> torch.Tensor:
        """Return event-level data probabilities"""
        logits = self.forward(batch)
        return logits.sigmoid()


class PartClassifier(Classifier):

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.z, c=batch.cond_z, mask=batch.mask_z)

        return self.net(batch.z)


from src.utils.utils import load_model
from omegaconf import DictConfig


class PsiClassifier(Classifier):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Load frozen ensemble
        # The base Classifier.__init__ instantiates self.net from cfg.net.
        # We want to replace it with the loaded ensemble.

        # We need to make sure we don't double-instantiate or if we do, we overwrite it.
        # super().__init__ calls instantiate(cfg.net).

        # Overwrite self.net with the loaded ensemble
        # Assuming cfg.cls_path points to the checkpoint directory of the ensemble
        self.net, _ = load_model(self.cfg.cls_path, Classifier, freeze=True)

        # Ensure it is ensembled
        assert self.net.ensembled, "Loaded classifier must be an ensemble"
        self.n_members = self.net.ensembled

        # Initialize psi
        self.psi = torch.nn.Parameter(torch.ones(self.n_members) / self.n_members)

        # We behave as a single classifier for the outside world (one prediction per sample)
        self.ensembled = False

    @property
    def trainable_parameters(self):
        return [self.psi]

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the combined log-likelihood ratio"""

        # Get ensemble logits: [E, B, 1] or [E, B] ?
        # Classifier.forward returns self.net(x).
        # If self.net is Model (Classifier) -> it calls Classifier.forward -> net(x)
        # If loaded model is Classifier, forward returns whatever its net returns.

        # We need to be careful. load_model returns a Classifier instance.
        # Calling self.net(batch) calls Classifier.forward(batch).
        # If that Classifier is ensembled, it returns [..., E] or [E, ...] depending on implementation?
        # Typically ensembled models in this codebase seem to return stacked outputs.
        # Let's verify Classifier.forward again.
        # It calls self.net(batch.x).

        # If self.net is the LOADED Classifier object:
        # logits = self.net(batch) -> returns [E, B] or [E, B, 1]?
        # In Unfolder, user used: lw_x = self.classifier(batch).squeeze(-1) -> implies [E, B, 1] -> [E, B]

        logits = self.net(batch)  # [E, B, 1]

        # Squeeze last dim
        logits = logits.squeeze(-1)  # [E, B]

        # Combine using psi (softmax)
        # psi: [E]
        # logits: [E, B]
        # output: [B]

        # logits_combined = torch.einsum('e, eb -> b', self.psi.softmax(0), logits)
        logits_combined = torch.einsum("e, eb -> b", self.psi, logits)

        # Unsqueeze to return [B, 1] to match expected output of forward()
        return logits_combined.unsqueeze(-1)
