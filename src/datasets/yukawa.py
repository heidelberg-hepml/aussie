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
from src.utils.transforms import (
    YukawaTransform,
    YukawaEquiTransform,
    YukawaAcceptanceTransform,
)


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

                if k == "x":  # limit number of reco jets to 8
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
        YukawaTransform(
            # fmt: off
            shift_x=torch.tensor([1.2,  8.3, 0., 0.]),
            scale_x=torch.tensor([4.7, 1.2, 1.2, 1.8]),
            shift_z=torch.tensor([9., 0., 0.]),
            scale_z=torch.tensor([1.6, 2.1, 1.8]),
            # fmt: on
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


@dataclass
class YukawaEquiProcess:

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


@dataclass
class YukawaAcceptanceProcess:

    num_features: int = 4
    transforms: Tuple[Callable] = (YukawaAcceptanceTransform(scale=450.0),)
    observables_x: Tuple[Observable] = tuple(
        momenta_to_observables(
            particle_names=["t", "h", "j"],
            delta_pairs=[(0, 1), (1, 2), (0, 2)],
            hard_scattering=True,
            off_shell=[False, False, False],
        )
    )
    observables_z: Tuple[Observable] = tuple()


# class LoThjProcess(Process):
#     def __init__(self, params: dict, device: torch.device):
#         self.params = params
#         self.analysis_as_test = params.get("analysis_as_test", False)
#         self.final_state = params["final_state"]
#         self.data = {}
#         self.device = device

#         self.importance_sampler = params.get("importance_sampler", False)

#     def load_data(self, subset: str):
#         """
#         Load training, validation, testing and analysis data from the specified h5 files
#         if it is not loaded yet

#         Args:
#             subset: Which part of the data, e.g. "train", "val", "test", "analysis"
#         """
#         if subset in self.data:
#             return

#         if subset in ["analysis", "analysis_eval"]:
#             path = self.params["analysis_file"]
#         else:
#             path = self.params["training_file"]
#         raw_data = pd.read_hdf(path).to_numpy()#[:300000]

#         n_events = len(raw_data)
#         alpha = torch.tensor(raw_data[:, :1], dtype=torch.float32, device=self.device)
#         x_hard = torch.tensor(
#             raw_data[:, 1:13].reshape(-1, 3, 4), dtype=torch.float32, device=self.device
#         )

#         x_reco = torch.tensor(
#             raw_data[:, 13:].reshape(n_events, -1, 4),
#             dtype=torch.float32,
#             device=self.device
#         )

#         max_reco_momenta = self.params.get("max_reco_momenta")
#         if max_reco_momenta is not None:
#             x_reco = x_reco[:, :max_reco_momenta]

#         min_reco_momenta = self.params.get("min_reco_momenta")
#         if min_reco_momenta is not None:
#             mask = (x_reco[:, :, 0] != 0.).bool().sum(dim=1) >= min_reco_momenta
#             x_reco = x_reco[mask]
#             x_hard = x_hard[mask]
#             alpha = alpha[mask]
#             n_events = len(x_reco)

#         if self.importance_sampler:
#             x_reco, x_hard = x_hard, x_reco

#         #torch.manual_seed(42)
#         perm = torch.randperm(len(x_hard))
#         x_hard = x_hard[perm]
#         x_reco = x_reco[perm]
#         alpha = alpha[perm]
#         #torch.manual_seed(torch.seed())

#         if subset == "analysis":
#             n_events = len(x_hard)
#             low, high = self.params["analysis_slice"]
#             data_slice = slice(int(n_events * low), int(n_events * high))
#             self.data["analysis"] = ProcessData(
#                 x_hard[data_slice],
#                 x_reco[data_slice],
#                 alpha=alpha[data_slice]
#             )
#         elif subset == "analysis_eval":
#             self.data["analysis_eval"] = ProcessData(x_hard, x_reco, alpha=alpha)

#         else:
#             for subs in ["train", "test", "val"]:
#                 low, high = self.params[f"{subs}_slice"]
#                 data_slice = slice(int(n_events * low), int(n_events * high))
#                 self.data[subs] = ProcessData(
#                     x_hard[data_slice], x_reco[data_slice], alpha=alpha[data_slice],
#                 )


#     def get_data(self, subset: str) -> ProcessData:
#         """
#         Returns data from the specified subset of the dataset.

#         Args:
#             subset: Which part of the data, e.g. "train", "val", "test", "analysis"
#         Returns:
#             ProcessData object containing the data
#         """
#         if subset in ["train", "val", "test", "analysis", "analysis_eval"]:
#             self.load_data(subset)
#             return self.data[subset]
#         else:
#             return self.data["test"]
#             #raise ValueError(f"Unknown subset '{subset}'")


#     def hard_masses(self) -> list[Optional[float]]:
#         """
#         Returns masses or None (if off-shell) for the hard-scattering level particles

#         Returns:
#             List of masses or None
#         """
#         return [173.2, 125.0, 0.]

#     def reco_masses(self) -> list[Optional[float]]:
#         """
#         Returns masses or None (if off-shell) for the reco-level particles

#         Returns:
#             List of masses or None
#         """
#         if self.final_state == "leptonic":
#             return [0., 0., 0., None, None] + [None]*(self.params.get("max_reco_momenta", 5) - 5)
#         else:
#             return [0., 0., None, None, None, None] + [None]*(self.params.get("max_reco_momenta", 6) - 6)

#     def truth_observables(self) -> list[Observable]:
#         """
#         Returns observables at the hard-scattering level for this process.

#         Returns:
#             List of observables
#         """
#         return momenta_to_observables(
#             particle_names=["t", "h", "j"],
#             delta_pairs=[(0, 1), (1, 2), (0, 2)],
#             hard_scattering=True,
#             off_shell=[False, False, False]
#         )

#     def reco_observables(self) -> list[Observable]:
#         """
#         Returns observables at the reconstruction level for this process.

#         Returns:
#             List of observables
#         """
#         reco_masses = self.reco_masses()
#         n_extra_jets = len(reco_masses) - 6
#         particle_names = (
#             [r"\gamma_1", r"\gamma_2", r"j_b", r"j_1", r"j_2", r"j_3"]
#             + [f"j_{{{i}}}" for i in range(4, 4 + n_extra_jets)]
#         )
#         delta_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

#         return momenta_to_observables(
#             particle_names=particle_names,
#             delta_pairs=delta_pairs,
#             hard_scattering=False,
#             off_shell=[m is None for m in reco_masses]
#         )

#     def observables(self) -> list[Observable]:
#         if not self.params.get("importance_sampler", False):
#             return self.reco_observables()
#         else:
#             return self.truth_observables()


# def momenta_to_observables(
#     particle_names: list[str],
#     delta_pairs: list[tuple[int, int]],
#     hard_scattering: bool,
#     off_shell: list[bool],
#     n_bins: int = 50,
#     sqrt_s: float = 13000.0,
# ) -> list[Observable]:
#     observables = []
#     for i, name in enumerate(particle_names):
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 0]),
#                 tex_label=f"E_{{{name}}}",
#                 unit="GeV",
#                 bins=lambda obs: get_quantile_bins(
#                     obs, n_bins=n_bins, lower=1e-4, upper=1e-4
#                 ),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 1]),
#                 tex_label=f"p_{{x,{name}}}",
#                 unit="GeV",
#                 bins=lambda obs: get_quantile_bins(
#                     obs, n_bins=n_bins, lower=1e-3, upper=1e-3
#                 ),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 2]),
#                 tex_label=f"p_{{y,{name}}}",
#                 unit="GeV",
#                 bins=lambda obs: get_quantile_bins(
#                     obs, n_bins=n_bins, lower=1e-3, upper=1e-3
#                 ),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: nanify(p[..., i, :], p[..., i, 3]),
#                 tex_label=f"p_{{z,{name}}}",
#                 unit="GeV",
#                 bins=lambda obs: get_quantile_bins(
#                     obs, n_bins=n_bins, lower=1e-3, upper=1e-3
#                 ),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: compute_pt(p[..., i, :]),
#                 tex_label=f"p_{{T,{name}}}",
#                 unit="GeV",
#                 bins=lambda obs: get_quantile_bins(obs, n_bins=n_bins, lower=1e-4, upper=1e-4),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: compute_phi(p[..., i, :]),
#                 tex_label=f"\\phi_{{{name}}}",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(-np.pi, np.pi, n_bins + 1),
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i: compute_eta(p[..., i, :]),
#                 tex_label=f"\\eta_{{{name}}}",
#                 unit=None,
#                 bins=lambda obs: get_quantile_bins(
#                     obs, n_bins=n_bins, lower=1e-4, upper=1e-4
#                 )
#             )
#         )
#         if off_shell[i]:
#             observables.append(
#                 Observable(
#                     compute=lambda p, i=i: compute_m(p[..., i, :]),
#                     tex_label=f"M_{{{name}}}",
#                     unit="GeV",
#                     bins=lambda obs: get_quantile_bins(obs, n_bins=n_bins, lower=1e-4, upper=1e-3),
#                 )
#             )

#     for i, j in delta_pairs:
#         name_i, name_j = particle_names[i], particle_names[j]
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i, j=j: (
#                     compute_phi(p[..., i, :]) - compute_phi(p[..., j, :]) + np.pi
#                 )
#                 % (2 * np.pi)
#                 - np.pi,
#                 tex_label=f"\\Delta \\phi_{{{name_i},{name_j}}}",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(-np.pi, np.pi, n_bins + 1),
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i, j=j: compute_eta(p[..., i, :])
#                 - compute_eta(p[..., j, :]),
#                 tex_label=f"\\Delta \\eta_{{{name_i},{name_j}}}",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(-6, 6, n_bins + 1),
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p, i=i, j=j: torch.sqrt(
#                     (
#                         (
#                             compute_phi(p[..., i, :])
#                             - compute_phi(p[..., j, :])
#                             + np.pi
#                         )
#                         % (2 * np.pi)
#                         - np.pi
#                     )
#                     ** 2
#                     + (compute_eta(p[..., i, :]) - compute_eta(p[..., j, :])) ** 2
#                 ),
#                 tex_label=f"\\Delta R_{{{name_i},{name_j}}}",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(0, 6, n_bins + 1),
#             )
#         )

#     if hard_scattering:
#         observables.append(
#             Observable(
#                 compute=lambda p: (p[..., 0].sum(dim=-1) + p[..., 3].sum(dim=-1))
#                 / sqrt_s,
#                 tex_label="x_1",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(0, 0.8, n_bins + 1),
#                 yscale="log",
#             )
#         )
#         observables.append(
#             Observable(
#                 compute=lambda p: (p[..., 0].sum(dim=-1) - p[..., 3].sum(dim=-1))
#                 / sqrt_s,
#                 tex_label="x_2",
#                 unit=None,
#                 bins=lambda obs: torch.linspace(0, 0.8, n_bins + 1),
#                 yscale="log",
#             )
#         )

#     return observables


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
