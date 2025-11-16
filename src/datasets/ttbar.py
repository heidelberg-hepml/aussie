import numpy as np
import os
import torch

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import ttbarTransform
from src.utils.utils import cartesian2jet


@tensorclass
class ttbarData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        m_dat: Optional[int] = 1695,
        m_sim: Optional[int] = 1725,
        device: Optional[torch.device] = None,
    ):

        assert m_dat != m_sim, "Choose different masses for Sim and Data"

        # read tensors into memory
        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, m in enumerate((m_sim, m_dat)):
            for k, prefix in zip(("x", "z"), ("rec", "gen")):

                arr = np.load(os.path.join(path, f"{prefix}_{m}_delphes.npy"))
                arr = cartesian2jet(arr.reshape(-1, 3, 4))

                tensor = torch.from_numpy(arr).float()
                tensor_kwargs[k].append(tensor)

            size = len(tensor)
            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k])

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@dataclass
class ttbarProcess:

    num_features: int = 16
    transforms: Tuple[Callable] = (
        ttbarTransform(
            # fmt: off
            shift_x=torch.tensor([2.95, 5.58, 0.0, 0.0, 13.64, 147.52, 0.0, 0.0, 9.24, 80.88, 0.0, 0.0, 4.65, 4.56, 4.36, 5.14,]),
            scale_x=torch.tensor([0.43, 0.25, 0.87, 1.81, 6.00, 47.83, 0.90, 1.81, 4.17, 34.36, 0.93, 1.81, 0.31, 0.32, 0.35, 0.18,]),
            shift_z=torch.tensor([2.93, 5.61, 0.00, 0.00, 13.81, 153.26, 0.00, 0.00, 9.57, 84.52, 0.00, 0.00, 4.68, 4.59, 4.40, 5.17,]),
            scale_z=torch.tensor([0.45, 0.26, 0.87, 1.82, 6.24, 49.72, 0.90, 1.82, 4.28, 35.81, 0.94, 1.81, 0.31, 0.32, 0.35, 0.18,]),
            # fmt: on
        ),
    )
    observables: Tuple[Observable] = (
        Observable(
            name="m_1",
            compute=lambda x: x[:, 0, 0],
            label=r"$m_1$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="pt_1",
            compute=lambda x: x[:, 0, 1],
            label=r"$p_{T,1}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="eta_1",
            compute=lambda x: x[:, 0, 2],
            label=r"$\eta_1$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="phi_1",
            compute=lambda x: x[:, 0, 3],
            label=r"$\phi_1$",
            xlims=(-3.5, 3.5),
        ),
        Observable(
            name="m_2",
            compute=lambda x: x[:, 1, 0],
            label=r"$m_2$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="pt_2",
            compute=lambda x: x[:, 1, 1],
            label=r"$p_{T,2}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="eta_2",
            compute=lambda x: x[:, 1, 2],
            label=r"$\eta_2$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="phi_2",
            compute=lambda x: x[:, 1, 3],
            label=r"$\phi_2$",
            xlims=(-3.5, 3.5),
        ),
        Observable(
            name="m_3",
            compute=lambda x: x[:, 2, 0],
            label=r"$m_3$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="pt_3",
            compute=lambda x: x[:, 2, 1],
            label=r"$p_{T,3}$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="eta_3",
            compute=lambda x: x[:, 2, 2],
            label=r"$\eta_3$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="phi_3",
            compute=lambda x: x[:, 2, 3],
            label=r"$\phi_3$",
            xlims=(-3.5, 3.5),
        ),
        Observable(
            name="M_12",
            compute=lambda p: compute_inv_mass(p, [0, 1]),
            label=r"$M_{12}$",
            qlims=(1e-2, 1 - 1e-2),
        ),
        Observable(
            name="M_13",
            compute=lambda p: compute_inv_mass(p, [0, 2]),
            label=r"$M_{13}$",
            qlims=(1e-2, 1 - 1e-2),
        ),
        Observable(
            name="M_23",
            compute=lambda p: compute_inv_mass(p, [1, 2]),
            label=r"$M_{23}$",
            qlims=(1e-2, 1 - 1e-2),
        ),
        Observable(
            name=r"M_123",
            compute=lambda p: compute_inv_mass(p, [0, 1, 2]),
            label=r"$M_{123}$",
            xlims=(100, 250),
        ),
    )

def compute_inv_mass(p, particles) -> torch.Tensor:

    px_sum = 0
    py_sum = 0
    pz_sum = 0
    e_sum = 0
    for particle in particles:

        m = p[..., particle, 0]
        pT = p[..., particle, 1]
        eta = p[..., particle, 2]
        phi = p[..., particle, 3]

        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        e = torch.sqrt(m**2 + px**2 + py**2 + pz**2)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    m = torch.sqrt(
        torch.clamp((e_sum) ** 2 - (px_sum) ** 2 - (py_sum) ** 2 - (pz_sum) ** 2, min=0)
    )
    return m

# def compute_inv_mass(p, particles) -> np.array:

#     px_sum = 0
#     py_sum = 0
#     pz_sum = 0
#     e_sum = 0
#     for particle in particles:

#         m = p[..., particle, 0]
#         pT = p[..., particle, 1]
#         eta = p[..., particle, 2]
#         phi = p[..., particle, 3]

#         px = pT * np.cos(phi)
#         py = pT * np.sin(phi)
#         pz = pT * np.sinh(eta)
#         e = np.sqrt(m**2 + px**2 + py**2 + pz**2)

#         px_sum += px
#         py_sum += py
#         pz_sum += pz
#         e_sum += e

#     m = np.sqrt(
#         np.clip(
#             (e_sum) ** 2 - (px_sum) ** 2 - (py_sum) ** 2 - (pz_sum) ** 2,
#             a_min=0,
#             a_max=None,
#         )
#     )
#     return m


# cartesian coords
# def inv_mass(p):
#     m2 = (2 * p[..., [0]] ** 2 - p**2).sum(-1).clamp(min=0)
#     return m2.sqrt()
