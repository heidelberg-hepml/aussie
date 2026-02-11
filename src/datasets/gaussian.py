import math
import torch
import torch.distributions as td

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable


@tensorclass
class GaussianToyData(UnfoldingData):

    @classmethod
    def read(
        cls,
        smear_width: float = 3,
        mu_sim=0.0,
        mu_dat=0.2,
        sigma_sim=1.0,
        sigma_dat=0.9,
        seed=69115,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):

        # rng
        gen = torch.Generator(device="cpu")

        params = [
            # (torch.tensor(mu_sim), torch.tensor(sigma_sim)),
            # (torch.tensor(mu_dat), torch.tensor(sigma_dat)),
            (mu_sim, sigma_sim),
            (mu_dat, sigma_dat),
        ]

        # px_sim = td.Normal(mu_sim, math.sqrt(sigma_sim**2 + smear_width**2))
        # px_dat = td.Normal(mu_dat, math.sqrt(sigma_dat**2 + smear_width**2))

        tensor_kwargs = defaultdict(list)
        for i, p in enumerate(params):

            # sample z with seed
            gen.manual_seed(seed + i)
            z = torch.normal(*p, size=[num, 1], generator=gen)

            # sample x | z with seed
            gen.manual_seed(seed + 2 + i)
            x = z + torch.normal(0, smear_width, size=[num, 1], generator=gen)

            tensor_kwargs["z"].append(z.float())
            tensor_kwargs["x"].append(x.float())
            tensor_kwargs["labels"].append(torch.full_like(z, i).squeeze())

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k])

        # return tensorclass dataset
        return cls(batch_size=[2 * num], device=device, **tensor_kwargs)


@dataclass
class GaussianToyProcess:
    num_features: int = 1
    transforms: Tuple[Callable] = ()
    observables_x: Tuple[Observable] = (
        Observable(
            name="gaussian",
            compute=lambda x: x[..., 0],
            label=r"$x$",
            qlims=(1e-2, 1 - 1e-2),
        ),
    )
    observables_z: Tuple[Observable] = (
        Observable(
            name="gaussian",
            compute=lambda z: z[..., 0],
            label=r"$z$",
            qlims=(1e-2, 1 - 1e-2),
        ),
    )
