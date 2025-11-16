from typing import Optional, Callable, Tuple
from dataclasses import dataclass
import torch


@dataclass
class Observable:
    """
    Data class for an observable used for plotting
    Args:
        name: Plain text name of the Observable
        compute: Function that computes the observable value for the given momenta
        label: Observable name for labels in plots
        bins: function that returns tensor with bin boundaries for given observable data
        logx: whether the X axis should be log-scaled
        logy: whether the Y axis should be log-scaled
        unit: Unit of the observable or None, if dimensionless, optional
    """

    name: str
    compute: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
    label: str
    qlims: Optional[Tuple[float]] = None
    xlims: Optional[Tuple[float]] = None
    logx: bool = False
    logy: bool = False
    discrete: bool = False
    unit: Optional[str] = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["compute"] = None
        d["bins"] = None
        return d
