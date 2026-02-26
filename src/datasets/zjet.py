import importlib
import numpy as np
import os
import torch
import warnings

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import ShiftAndScale


@tensorclass
class ZJetData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
        correction_weights: Optional[str] = None,
        exclusions=("particles", "Zs", "lhas"),
        data_key: str = "Herwig",
    ):

        ef = importlib.import_module("energyflow")

        # read tensors into memory
        load_kwargs = dict(num_data=num or -1, exclude_keys=exclusions, cache_dir=path)

        dicts = [
            ef.zjets_delphes.load("Pythia21", **load_kwargs),  # "Sim"
            ef.zjets_delphes.load(data_key, **load_kwargs),  # "Data"
        ]

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, d in enumerate(dicts):

            if correction_weights is not None:  # correct for hidden-variable effects
                d = dicts[0]  # always select Pythia

            mask = (  # avoid divide by zero and log(zero)
                (d[f"gen_widths"] != 0)
                & (d[f"sim_widths"] != 0)
                & (d[f"gen_sdms"] > 0)
                & (d[f"sim_sdms"] > 0)
            )

            for k, prefix in zip(("x", "z"), ("sim", "gen")):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    # fmt: off
                    tensor = torch.from_numpy(
                        np.stack([
                            d[f"{prefix}_jets"][:, -1] ,                                 # mass
                            d[f"{prefix}_mults"],                                        # mult
                            d[f"{prefix}_widths"],                                       # width
                            d[f"{prefix}_tau2s"] / d[f"{prefix}_widths"],                # tau_21
                            2 * np.log(d[f"{prefix}_sdms"] / d[f"{prefix}_jets"][:, 0]), # ln_rho
                            d[f"{prefix}_zgs"],                                          # zg
                        ], axis=1)[mask]
                    ).float()
                    # fmt: on
                size = len(tensor)
                tensor_kwargs[k].append(tensor)

            # construct correction weights for data
            tensor_kwargs["sample_logweights"].append(
                torch.from_numpy(np.load(correction_weights)[:size]).float().log()
                if (correction_weights and i)
                else torch.full([size], 0, dtype=torch.float32)
            )

            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k])

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@tensorclass
class ZjetTransform:
    """
    TODO
    """

    shift_x: torch.Tensor
    shift_z: torch.Tensor
    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-3

    def forward(self, batch):

        # process reco level
        batch.x[:, 1] += torch.rand_like(batch.x[:, 1]) - 0.5
        batch.x[:, :4] = batch.x[:, :4].add(self.eps).log()
        # batch.x[:, [0, 1, 2, 3, 6]] = batch.x[:, [0, 1, 2, 3, 6]].add(self.eps).log()
        batch.x = ShiftAndScale.forward(batch.x, shift=self.shift_x, scale=self.scale_x)

        # process part level
        batch.z[:, 1] += torch.rand_like(batch.z[:, 1]) - 0.5
        batch.z[:, :4] = batch.z[:, :4].add(self.eps).log()
        # batch.z[:, [0, 1, 2, 3, 6]] = batch.z[:, [0, 1, 2, 3, 6]].add(self.eps).log()
        batch.z = ShiftAndScale.forward(batch.z, shift=self.shift_z, scale=self.scale_z)

        return batch

    def reverse(self, batch):

        # process reco level
        batch.x = ShiftAndScale.reverse(batch.x, shift=self.shift_x, scale=self.scale_x)
        batch.x[:, :4] = batch.x[:, :4].exp().sub(self.eps)
        batch.x[:, 1] = batch.x[:, 1].round()

        # process part level
        batch.z = ShiftAndScale.reverse(batch.z, shift=self.shift_z, scale=self.scale_z)
        batch.z[:, :4] = batch.z[:, :4].exp().sub(self.eps)
        batch.z[:, 1] = batch.z[:, 1].round()

        return batch


@dataclass
class ZJetProcess:
    num_features: int = 6
    transforms: Tuple[Callable] = (
        ZJetTransform(
            shift_x=torch.tensor([2.7727, 2.8380, -2.0688, -0.4431, -6.7655, 0.2401]),
            scale_x=torch.tensor([0.5247, 0.4587, 0.6379, 0.3688, 2.3824, 0.1188]),
            shift_z=torch.tensor([2.9788, 3.1999, -2.2242, -0.3680, -7.0247, 0.2337]),
            scale_z=torch.tensor([0.4525, 0.3994, 0.6832, 0.3327, 2.3020, 0.1152]),
        ),
    )
    observables_x: Tuple[Observable] = (
        Observable(
            name="mass",
            compute=lambda x: x[..., 0],
            label=r"$\text{Jet mass } m$",
            logy=True,
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda x: x[..., 1],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda x: x[..., 2],
            label=r"$\text{Jet width } w$",
            logy=True,
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda x: x[..., 3],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            xlims=(0.1, 1.1),
        ),
        Observable(
            name="log_rho",
            compute=lambda x: x[..., 4],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
        Observable(
            name="zg",
            compute=lambda x: x[..., 5],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
    )
    observables_z: Tuple[Observable] = (
        Observable(
            name="mass",
            compute=lambda z: z[..., 0],
            label=r"$\text{Jet mass } m$",
            logy=True,
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda z: z[..., 1],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda z: z[..., 2],
            label=r"$\text{Jet width } w$",
            logy=True,
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda z: z[..., 3],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            xlims=(0.1, 1.1),
        ),
        Observable(
            name="log_rho",
            compute=lambda z: z[..., 4],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
        Observable(
            name="zg",
            compute=lambda z: z[..., 5],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
    )
