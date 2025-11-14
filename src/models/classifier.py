import torch
import torch.nn.functional as F

from src.models.base_model import Model
from src.networks import TransformerEncoder
from src.utils.datasets import StationData


class Classifier(Model):

    @property
    def lowlevel(self):
        return isinstance(self.net, TransformerEncoder)

    def batch_loss(self, batch: StationData) -> torch.Tensor:

        x = torch.cat([batch.x_dat, batch.x_sim])

        logits = self.forward(x).squeeze(-1)

        labels = torch.zeros_like(logits)
        labels[: len(batch.x_dat)] = 1

        match self.cfg.loss:
            case "bce":
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
            case "mlc":
                sign = 1 - 2 * labels
                z = sign * logits
                loss = z + z.exp()
            case "mse":
                sign = 2 * labels - 1
                z = sign * logits
                loss = (-2 * z).exp() - 2 * z.exp()
            case _:
                raise ValueError

        # if batch.data_weights is not None:
        #     loss = loss * batch.data_weights

        loss = loss.mean(0)
        if self.bayesian:
            kld = self.kld / self.train_size
            self.log_scalar(kld, "KL")
            self.log_scalar(loss, "loss_raw")
            loss = loss + kld

        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reco-level data-to-sim log-likelihood ratio"""
        return self.net(x)

    def weight(self, x: torch.Tensor) -> torch.Tensor:
        """Return the event-level data-to-sim likelihood ratio"""
        logits = self.forward()
        return logits.exp()

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return event-level data probabilities"""
        logits = self.forward(x)
        return logits.sigmoid()
