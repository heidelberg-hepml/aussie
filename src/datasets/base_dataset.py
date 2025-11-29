import torch

from abc import abstractmethod
from tensordict import tensorclass
from typing import Optional

@tensorclass
class UnfoldingData:
    """
    A tensorclass for holding unfolding data.
    The optional attributes are
        - "x"      : Reco-level features
        - "z"      : Particle/parton-level features
        - "labels" : Binary labels (Data=1, Sim=0)
    """

    x: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None
    aux_x: Optional[torch.Tensor] = None
    aux_z: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

    @classmethod
    @abstractmethod
    def read_omnifold(cls, path: str, device: Optional[torch.device] = None):
        pass