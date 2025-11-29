import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.networks import TransformerEncoder
from src.datasets import UnfoldingData

class Classifier(Model):

    @property
    def lowlevel(self):
        return isinstance(self.net, TransformerEncoder)

    def batch_loss(self, batch: UnfoldingData) -> torch.Tensor:

        # forward pass
        logits = self.forward(batch).squeeze(-1)

        # split by labels to handle imbalanced classes
        logits_sim = logits[..., batch.labels == 0]  # ensembling
        logits_dat = logits[..., batch.labels == 1]  # ensembling

        mean_dims = (0, 1) if self.ensembled else 0
        match self.cfg.loss:

            case "bce":
                loss = (
                    F.binary_cross_entropy_with_logits(
                        logits_dat, torch.ones_like(logits_dat), reduction="none"
                    ).mean(mean_dims)
                    + F.binary_cross_entropy_with_logits(
                        logits_sim, torch.zeros_like(logits_sim), reduction="none"
                    ).mean(mean_dims)
                ) / 2
            case "mlc":
                loss = (
                    (logits_sim + logits_sim.exp()).mean(mean_dims)
                    + (-logits_dat + (-logits_dat).exp()).mean(mean_dims)
                ) / 2
            case _:
                raise ValueError

        # if batch.data_weights is not None:
        #     loss = loss * batch.data_weights

        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        return loss

    def forward(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""

        if self.lowlevel:
            return self.net(batch.x, mask=batch.x[..., 0] != 0)

        return self.net(batch.x)

    def weight(self, batch: UnfoldingData) -> torch.Tensor:
        """Return the event-level data-to-sim likelihood ratio"""
        logits = self.forward(batch)
        return logits.exp()

    def prob(self, batch: UnfoldingData) -> torch.Tensor:
        """Return event-level data probabilities"""
        logits = self.forward(batch)
        return logits.sigmoid()
