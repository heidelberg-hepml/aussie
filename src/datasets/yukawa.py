import pandas as pd
import numpy as np
import os
import torch

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import LorentzTransform


@tensorclass
class YukawaData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        data_angle: float = 45.0,
        device: Optional[torch.device] = None,
        num: Optional[int] = None,
    ):

        dsets = [
            pd.read_hdf(os.path.join(path, f), stop=num).to_numpy()
            for f in (
                "nompi_sampled_new.h5",  # Sim
                "cp_vio_yukawa_nompi_fixed.h5",  # Data
            )
        ]

        # read tensors into memory
        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, d in enumerate(dsets):

            for k, start, stop in zip(("x", "z"), (13, 1), (None, 13)):

                # pick angle for data
                if i:
                    d = d[d[..., 0] == data_angle]

                momenta = d[..., start:stop].reshape(d.shape[0], -1, 4)

                if k == "x":  # limit number of reco objects to 8
                    momenta = momenta[:, :8]

                tensor = torch.from_numpy(momenta).float()

                # mask out empty particles
                mask = tensor[..., 0] > 0
                tensor_kwargs[k].append(tensor)
                tensor_kwargs[f"mask_{k}"].append(mask)

            size = len(tensor)
            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))
            tensor_kwargs["conds"].append(torch.from_numpy(d[..., 0]).float())

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k])

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@tensorclass
class YukawaTransform:
    """
    TODO
    """

    scale_x: torch.Tensor
    scale_z: torch.Tensor
    eps: float = 1e-4

    def forward(self, batch):

        batch.x /= self.scale_x
        batch.z /= self.scale_z

        # create particle ids
        ids_x = torch.eye(4, device=batch.device)[
            None, [0, 1, 2, 3, 3, 3, 3, 3], :
        ].expand(len(batch), -1, -1)
        ids_z = torch.eye(3, device=batch.device)[None, [0, 1, 2], :].expand(
            len(batch), -1, -1
        )
        batch.x = torch.cat([batch.x, ids_x], dim=2)
        batch.z = torch.cat([batch.z, ids_z], dim=2)

        batch.x, batch.mask_x = LorentzTransform.forward(batch.x, mask=batch.mask_x)
        batch.z, batch.mask_z = LorentzTransform.forward(batch.z, mask=batch.mask_z)

        return batch

    def reverse(self, batch):
        raise NotImplementedError
        return batch


def momenta_to_observables(
    particle_names: list[str],
    delta_pairs: list[tuple[int, int]],
    hard_scattering: bool,
    off_shell: list[bool],
    sqrt_s: float = 13000.0,
) -> list[Observable]:
    observables = []
    for i, name in enumerate(particle_names):
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 0]),
                label=f"$E_{{{name}}}$",
                unit="GeV",
                qlims=(1e-4, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 1]),
                label=f"$p_{{x,{name}}}$",
                unit="GeV",
                qlims=(1e-3, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 2]),
                label=f"$p_{{y,{name}}}$",
                unit="GeV",
                qlims=(1e-3, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 3]),
                label=f"$p_{{z,{name}}}$",
                unit="GeV",
                qlims=(1e-3, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: compute_pt(p[..., i, :]),
                label=f"$p_{{T,{name}}}$",
                unit="GeV",
                qlims=(1e-4, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: compute_phi(p[..., i, :]),
                label=f"$\\phi_{{{name}}}$",
                unit=None,
                xlims=(-np.pi, np.pi),
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i: compute_eta(p[..., i, :]),
                label=f"$\\eta_{{{name}}}$",
                unit=None,
                logy=True,
                qlims=(1e-3, 1 - 1e-3),
            )
        )
        if off_shell[i]:
            observables.append(
                Observable(
                    name="placeholder",
                    compute=lambda p, i=i: compute_m(p[..., i, :]),
                    label=f"$M_{{{name}}}$",
                    unit="GeV",
                    qlims=(1e-4, 1 - 1e-3),
                )
            )

    for i, j in delta_pairs:
        name_i, name_j = particle_names[i], particle_names[j]
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i, j=j: (
                    compute_phi(p[..., i, :]) - compute_phi(p[..., j, :]) + np.pi
                )
                % (2 * np.pi)
                - np.pi,
                label=f"$\\Delta \\phi_{{{name_i},{name_j}}}$",
                unit=None,
                xlims=(-np.pi, np.pi),
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i, j=j: compute_eta(p[..., i, :])
                - compute_eta(p[..., j, :]),
                label=f"$\\Delta \\eta_{{{name_i},{name_j}}}$",
                unit=None,
                # xlims=(6, 6),
                qlims=(1e-3, 1 - 1e-3),
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p, i=i, j=j: torch.sqrt(
                    (
                        (compute_phi(p[..., i, :]) - compute_phi(p[..., j, :]) + np.pi)
                        % (2 * np.pi)
                        - np.pi
                    )
                    ** 2
                    + (compute_eta(p[..., i, :]) - compute_eta(p[..., j, :])) ** 2
                ),
                label=f"$\\Delta R_{{{name_i},{name_j}}}$",
                unit=None,
                # xlims=(0, 6),
                qlims=(0, 1 - 1e-3),
            )
        )

    if hard_scattering:
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p: (p[..., 0].sum(dim=-1) + p[..., 3].sum(dim=-1))
                / sqrt_s,
                label="$x_1$",
                unit=None,
                # xlims=(0, 0.8),
                qlims=(1e-4, 1 - 1e-3),
                logy=True,
            )
        )
        observables.append(
            Observable(
                name="placeholder",
                compute=lambda p: (p[..., 0].sum(dim=-1) - p[..., 3].sum(dim=-1))
                / sqrt_s,
                label="$x_2$",
                unit=None,
                # xlims=(0, 0.8),
                qlims=(1e-4, 1 - 1e-3),
                logy=True,
            )
        )

    return observables


@dataclass
class YukawaProcess:

    num_features: int = 16
    transforms: Tuple[Callable] = (
        YukawaEquiTransform(
            scale_x=140.0,
            scale_z=450.0,
        ),
    )
    observables_z: Tuple[Observable] = tuple(
        momenta_to_observables(
            particle_names=["t", "h", "j"],
            delta_pairs=[(0, 1), (1, 2), (0, 2)],
            hard_scattering=True,
            off_shell=[False, False, False],
        )
    )
    observables_x: Tuple[Observable] = tuple(
        momenta_to_observables(
            particle_names=[r"\gamma_1", r"\gamma_2", r"j_b", r"j_1", r"j_2", r"j_3"]
            + [f"j_{{{i}}}" for i in range(4, 6)],
            delta_pairs=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            hard_scattering=False,
            off_shell=[0, 0] + [None] * 6,
        )
    )


def nanify(p: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    return torch.where(
        p[..., 0] != 0.0,
        obs,
        torch.tensor(float("nan"), device=p.device, dtype=p.dtype),
    )


def compute_pt(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2))


def compute_phi(p: torch.Tensor) -> torch.Tensor:
    return nanify(p, torch.arctan2(p[..., 2], p[..., 1]))


def compute_eta(p: torch.Tensor) -> torch.Tensor:
    return nanify(
        p,
        torch.arctanh(
            p[..., 3] / torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2 + p[..., 3] ** 2)
        ),
    )


def compute_m(p: torch.Tensor) -> torch.Tensor:
    return nanify(
        p,
        torch.sqrt(
            torch.clamp(
                p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2, min=0
            )
        ),
    )
