import importlib
import numpy as np
import torch

from collections import defaultdict
from dataclasses import dataclass
from tensordict import tensorclass
from typing import Callable, Tuple, Optional

from src.datasets.base_dataset import UnfoldingData
from src.utils.observable import Observable
from src.utils.transforms import OmniFoldTransform


@tensorclass
class OmniFoldData(UnfoldingData):

    @classmethod
    def read(
        cls,
        path: str,
        num: Optional[int] = None,
        device: Optional[torch.device] = None,
        exclusions=("particles", "Zs", "lhas"),
    ):

        ef = importlib.import_module("energyflow")

        # read tensors into memory
        load_kwargs = dict(num_data=num or -1, exclude_keys=exclusions, cache_dir=path)

        dicts = [
            ef.zjets_delphes.load("Pythia21", **load_kwargs),  # "Sim"
            ef.zjets_delphes.load("Herwig", **load_kwargs),  # "Data"
        ]
        features = ("ang2s", "mults", "sdms", "tau2s", "widths", "zgs")

        batch_size = 0
        tensor_kwargs = defaultdict(list)
        for i, d in enumerate(dicts):

            for k, prefix in zip(("x", "z"), ("sim", "gen")):

                tensor = torch.from_numpy(
                    np.stack([d[f"{prefix}_{f}"] for f in features], axis=1)
                ).float()
                size = len(tensor)

                tensor_kwargs[k].append(tensor)

            batch_size += size
            tensor_kwargs["labels"].append(torch.full([size], i, dtype=torch.float32))

        for k in tensor_kwargs:
            tensor_kwargs[k] = torch.cat(tensor_kwargs[k])

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


@dataclass
class OmniFoldProcess:
    num_features: int = 6
    transforms: Tuple[Callable] = (
        OmniFoldTransform(
            shift_x=torch.tensor([-3.2852, 2.8379, 1.9633, -2.5057, -2.0682, 0.2401]),
            scale_x=torch.tensor([0.9177, 0.4593, 1.1577, 0.5972, 0.6403, 0.1191]),
            shift_z=torch.tensor([-3.4062, 3.1988, 1.8654, -2.5909, -2.2248, 0.2337]),
            scale_z=torch.tensor([0.9155, 0.4019, 1.1282, 0.5534, 0.6859, 0.1154]),
        ),
    )
    observables: Tuple[Observable] = (
        Observable(
            name="ang2",
            compute=lambda x: x[..., 0],
            label=None,
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="mult",
            compute=lambda x: x[..., 1],
            label=None,
            discrete=2,
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="sdm",
            compute=lambda x: x[..., 2],
            label=None,
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="tau2",
            compute=lambda x: x[..., 3],
            label=None,
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="width",
            compute=lambda x: x[..., 4],
            label=None,
            qlims=(1e-3, 1 - 1e-3),
        ),
        Observable(
            name="zg",
            compute=lambda x: x[..., 5],
            label=None,
            xlims=(0.08, 0.52),
        ),
    )