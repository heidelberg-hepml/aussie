import importlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import warnings

from collections import defaultdict
from dataclasses import dataclass
from lgatr import get_spurions
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import OmniFoldCartesianTransform


@tensorclass
class OmniFoldCartesianData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        observables_path: str,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
        correction_weights: Optional[str] = None,
        data_key: str = "Herwig",
    ):

        ef = importlib.import_module("energyflow")

        # read tensors into memory
        load_kwargs = dict(num_data=num or -1, cache_dir=path, pad=True)

        dicts = [
            ef.zjets_delphes.load("Pythia21", **load_kwargs),  # "Sim"
            ef.zjets_delphes.load(data_key, **load_kwargs),  # "Data"
        ]

        observables = [
            np.load(os.path.join(observables_path, "zjet_observables_Pythia21.npz")),
            np.load(os.path.join(observables_path, f"zjet_observables_{data_key}.npz")),
        ]

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, (d, o) in enumerate(zip(dicts, observables)):

            if correction_weights is not None:  # correct for hidden-variable effects
                d = dicts[0]  # always select Pythia

            keep = (  # avoid divide by zero and log(zero)
                (o[f"sim_tau1"] != 0)
                & (o[f"gen_tau1"] != 0)
                & (o[f"sim_log_rho"] > -100)
                & (o[f"gen_log_rho"] > -100)
            )[:num]

            for k, prefix in zip(("x", "z"), ("sim", "gen")):

                # convert to (E, p)
                jets = d[f"{prefix}_jets"]
                particles = d[f"{prefix}_particles"]
                pids = particles[..., [-1]].copy()

                # unscale pts and uncenter eta, phi
                particles[..., 0] *= 100
                particles[..., [1, 2]] += jets[:, None, [1, 2]]

                # convert to E, px, py, pz
                particle_momenta = ef.p4s_from_ptyphipids(particles)
                particle_momenta *= particles[..., [0]] > 0

                # stack momenta with (scaled) pids
                features = np.dstack([particle_momenta, pids])
                tensor = torch.from_numpy(features).float()
                # boost into jet rest frame
                jet_momenta = torch.from_numpy(ef.p4s_from_ptyphims(jets)).float()

                # tensor = F.pad(tensor, (0, 0, 0, 159 - tensor.size(1)))
                tensor = F.pad(tensor, (0, 0, 0, 157 - tensor.size(1)))[keep]
                tensor_kwargs[k].append(tensor)
                tensor_kwargs[f"mask_{k}"].append(tensor[..., 0] > 0)
                tensor_kwargs[f"cond_{k}"].append(jet_momenta[keep])
                tensor_kwargs[f"aux_{k}"].append(  # for plotting
                    torch.from_numpy(
                        # fmt: off
                        np.stack([
                            o[f"{prefix}_jet_pt"][:num],       # pt
                            o[f"{prefix}_jet_mass"][:num],     # mass
                            o[f"{prefix}_multiplicity"][:num], # mult
                            o[f"{prefix}_tau1"][:num],         # width
                            o[f"{prefix}_tau21"][:num],        # nsub ratio
                            o[f"{prefix}_zg"][:num],           # zg
                            o[f"{prefix}_log_rho"][:num],      # sd mass
                        ], axis=1)
                        # fmt: on
                    )[keep].float()
                )


            size = len(tensor)

            # construct correction weights for data
            tensor_kwargs["sample_logweights"].append(
                torch.from_numpy(np.load(correction_weights)[:size]).float().log()
                if (correction_weights and i)
                else torch.full([size], 0, dtype=torch.float32)
            )

            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k], dim=0)

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@dataclass
class OmniFoldCartesianProcess:
    num_features: int = 4
    transforms: Tuple[Callable] = (OmniFoldCartesianTransform(),)
    observables_x: Tuple[Observable] = (
        Observable(
            name="pt",
            compute=lambda x: x[..., 0],
            label=r"$\text{Jet transverse momentum } p_T$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="mass",
            compute=lambda x: x[..., 1],
            label=r"$\text{Jet mass } m$",
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda x: x[..., 2],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda x: x[..., 3],
            label=r"$\text{Jet width } w$",
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda x: x[..., 4],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            # xlims=(0.1, 1.1),
            qlims=(1e-3, 1-1e-3),
        ),
        Observable(
            name="zg",
            compute=lambda x: x[..., 5],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
        Observable(
            name="log_rho",
            compute=lambda x: x[..., 6],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
    )
    observables_z: Tuple[Observable] = (
        Observable(
            name="pt",
            compute=lambda z: z[..., 0],
            label=r"$\text{Jet transverse momentum } p_T$",
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="mass",
            compute=lambda z: z[..., 1],
            label=r"$\text{Jet mass } m$",
            xlims=(1, 60),
        ),
        Observable(
            name="mult",
            compute=lambda z: z[..., 2],
            label=r"$\text{Jet multiplicity } N$",
            discrete=2,
            xlims=(0, 60),
        ),
        Observable(
            name="width",
            compute=lambda z: z[..., 3],
            label=r"$\text{Jet width } w$",
            xlims=(0, 0.6),
        ),
        Observable(
            name="tau_21",
            compute=lambda z: z[..., 4],
            label=r"$\text{N-subjettiness ratio } \tau_{21}$",
            # xlims=(0.1, 1.1),
            qlims=(1e-3, 1-1e-3),
        ),
        Observable(
            name="zg",
            compute=lambda z: z[..., 5],
            label=r"$\text{Groomed momentum fraction }z_g$",
            xlims=(0.08, 0.52),
        ),
        Observable(
            name="log_rho",
            compute=lambda z: z[..., 6],
            label=r"$\text{Groomed mass }\log \rho$",
            xlims=(-14, -2),
        ),
    )
