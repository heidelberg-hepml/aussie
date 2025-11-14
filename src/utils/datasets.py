import importlib
import numpy as np
import os
import torch

from tensordict import tensorclass
from energyflow import zjets_delphes  # TODO:  move to local scope
from typing import List, Optional

KEYS = ("x_sim", "z_sim", "x_dat", "z_dat")


@tensorclass
class StationData:
    """
    A tensorclass for holding unfolding data.
    The optional attributes are
        - "x_sim" : Simulated measurements
        - "z_sim" : Simulated latents
        - "x_dat" : Observed measurements
        - "x_dat" : Truth latents
    The following attributes are placeholders, filled internally by STATION.
        - "labels"                          (N_events, 1):                TODO: Description
        - "is_break"                        (N_events, 100):              TODO: Description
        - "accepted"                        (N_events, 100):              TODO: Description
        - "num_rej"                         (N_events,):                  TODO: Description
        - "in_chain_n"                      (N_events, 100, max_num_rej)  TODO: Description
        - "w_class"                         (N_events,):                  TODO: Description
        - "w_ref_break"                     (N_events, 100):              TODO: Description
        - "w_ref_chain"                     (N_events,):                  TODO: Description
        - "w_ref_history"                   (N_events,):                  TODO: Description
        - "w_ref_event"                     (N_events,):                  TODO: Description
        - "sample_weights"                  (N_events,):                  TODO: Description
    """

    x_sim: Optional[torch.Tensor] = None
    z_sim: Optional[torch.Tensor] = None
    x_dat: Optional[torch.Tensor] = None
    z_dat: Optional[torch.Tensor] = None

    @classmethod
    def read_omnifold(
        cls,
        path: str,
        num: Optional[int] = None,
        keys: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        exclusions=("particles", "Zs", "lhas"),
    ):

        # check keys
        keys = resolve_keys(keys)

        # read tensors into memory
        tensor_kwargs = {}
        load_kwargs = dict(num_data=num or -1, exclude_keys=exclusions, cache_dir=path)

        ef = importlib.import_module("energyflow")
        dict_sim = ef.zjets_delphes.load("Pythia21", **load_kwargs)
        if any(k.endswith("dat") for k in keys):
            dict_dat = ef.zjets_delphes.load("Herwig", **load_kwargs)

        for k in keys:
            prefix = "gen" if k.startswith("z") else "sim"
            arrs = dict_dat if k.endswith("dat") else dict_sim
            labels = ("ang2s", "mults", "sdms", "tau2s", "widths", "zgs")

            tensor = torch.from_numpy(
                np.stack([arrs[f"{prefix}_{l}"] for l in labels], axis=1)
            ).float()
            batch_size = len(tensor)

            tensor_kwargs[k] = tensor

        # return tensorclass dataset
        return cls(batch_size=[batch_size], device=device, **tensor_kwargs)


def resolve_keys(keys):

    # use all keys by default
    keys = keys or KEYS

    # check that all keys are known
    unknown_keys = set(keys) - set(KEYS)
    assert unknown_keys == set(), f"Found unknown keys {unknown_keys}"

    return keys
