import importlib
import numpy as np
import torch
import torch.nn.functional as F
import warnings

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import OmniFoldParticleTransform


@tensorclass
class OmniFoldParticleData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        use_cond: bool = False,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):

        ef = importlib.import_module("energyflow")

        # read tensors into memory
        load_kwargs = dict(num_data=num or -1, cache_dir=path, pad=True)

        dicts = [
            ef.zjets_delphes.load("Pythia21", **load_kwargs),  # "Sim"
            ef.zjets_delphes.load("Herwig", **load_kwargs),  # "Data"
        ]

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, d in enumerate(dicts):

            for k, prefix in zip(("x", "z"), ("sim", "gen")):

                tensor = torch.from_numpy(d[f"{prefix}_particles"]).float()
                # tensor = F.pad(tensor, (0, 0, 0, 152 - tensor.size(1)))
                tensor = F.pad(tensor, (0, 0, 0, 159 - tensor.size(1)))
                mask = tensor[..., 0] > 0
                tensor_kwargs[k].append(tensor)

                tensor_kwargs[f"mask_{k}"].append(mask)

                if use_cond:
                    tensor_kwargs[f"cond_{k}"].append(
                        torch.from_numpy(
                            # fmt: off
                            np.stack([
                                d[f"{prefix}_jets"][:, 0],                                   # pt
                                d[f"{prefix}_jets"][:, 1],                                   # eta
                                d[f"{prefix}_jets"][:, 2],                                   # phi
                                d[f"{prefix}_jets"][:, 3],                                   # mass
                            ], axis=1)
                            # fmt: on
                        ).float()
                    )

                tensor_kwargs[f"aux_{k}"].append(  # for plotting
                    torch.from_numpy(
                        # fmt: off
                        np.stack([
                            d[f"{prefix}_jets"][:, 0],                                   # pt
                            d[f"{prefix}_jets"][:, 1],                                   # eta
                            d[f"{prefix}_jets"][:, 2],                                   # phi
                            d[f"{prefix}_jets"][:, 3],                                   # mass
                            d[f"{prefix}_mults"],                                        # mult
                            d[f"{prefix}_widths"],                                       # width
                            d[f"{prefix}_tau2s"] / d[f"{prefix}_widths"],                # tau_21
                            2 * np.log(d[f"{prefix}_sdms"] / d[f"{prefix}_jets"][:, 0]), # ln_rho
                            d[f"{prefix}_zgs"],                                          # zg
                        ], axis=1)
                        # fmt: on
                    ).float()
                )

            size = len(tensor)
            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@dataclass
class OmniFoldParticleProcess:
    num_features: int = 4
    transforms: Tuple[Callable] = (
        OmniFoldParticleTransform(
            shift_x=torch.tensor([-2.9831, 0.0, 0.0, 0.0]),
            scale_x=torch.tensor([1.2479, 0.1128, 0.1169, 1.0]),
            shift_z=torch.tensor([-3.6219, 0.0, 0.0, 0.0]),
            scale_z=torch.tensor([1.6644, 0.1283, 0.1278, 1.0]),
        ),
    )
    observables: Tuple[Observable] = (
        Observable(
            name="pt",
            compute=lambda x: x[..., 0],
            label=r"$\text{Jet transverse momentum } p_T$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="eta",
            compute=lambda x: x[..., 1],
            label=r"$\text{Jet rapidity } \eta$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="phi",
            compute=lambda x: x[..., 2],
            label=r"$\text{Jet azimuthal angle } \phi$",
            xlims=(0, 2 * np.pi),
        ),
        Observable(
            name="mass",
            compute=lambda x: x[..., 3],
            label=r"$\text{Jet mass } m$",
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda x: x[..., 4],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda x: x[..., 5],
            label=r"$\text{Jet width } w$",
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda x: x[..., 6],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            xlims=(0.1, 1.1),
        ),
        Observable(
            name="log_rho",
            compute=lambda x: x[..., 7],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
        Observable(
            name="zg",
            compute=lambda x: x[..., 8],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
    )


@dataclass
class OmniFoldParticleCondProcess:
    num_features: int = 4
    transforms: Tuple[Callable] = (
        OmniFoldParticleTransform(
            shift_x=torch.tensor([-3.7743, 0.0, 0.0, 0.0]),
            scale_x=torch.tensor([1.2560, 0.1128, 0.1169, 1.0]),
            shift_z=torch.tensor([-4.4319, 0.0, 0.0, 0.0]),
            scale_z=torch.tensor([1.6635, 0.1283, 0.1278, 1.0]),
            shift_cond_x=torch.tensor([5.3502, 0.0, 3.1532, 2.7719]),
            scale_cond_x=torch.tensor([0.3751, 1.3483, 1.8122, 0.5310]),
            shift_cond_z=torch.tensor([5.3836, 0.0, 3.1528, 2.9777]),
            scale_cond_z=torch.tensor([0.3466, 1.3496, 1.8122, 0.4571]),
        ),
    )
    observables: Tuple[Observable] = (
        Observable(
            name="pt",
            compute=lambda x: x[..., 0],
            label=r"$\text{Jet transverse momentum } p_T$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="eta",
            compute=lambda x: x[..., 1],
            label=r"$\text{Jet rapidity } \eta$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="phi",
            compute=lambda x: x[..., 2],
            label=r"$\text{Jet azimuthal angle } \phi$",
            xlims=(0, 2 * np.pi),
        ),
        Observable(
            name="mass",
            compute=lambda x: x[..., 3],
            label=r"$\text{Jet mass } m$",
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda x: x[..., 4],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda x: x[..., 5],
            label=r"$\text{Jet width } w$",
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda x: x[..., 6],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            xlims=(0.1, 1.1),
        ),
        Observable(
            name="log_rho",
            compute=lambda x: x[..., 7],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
        Observable(
            name="zg",
            compute=lambda x: x[..., 8],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
    )
