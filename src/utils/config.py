import os
import sys
from omegaconf import OmegaConf, open_dict


def get_prev_config(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, ".hydra/config.yaml"))


def get_prev_overrides(prev_exp_dir):
    return OmegaConf.load(os.path.join(prev_exp_dir, ".hydra/overrides.yaml"))


def update_config_from_prev(cfg, hydra_cfg, prev_exp_dir):
    prev_cfg = get_prev_config(prev_exp_dir)
    overrides = OmegaConf.from_dotlist(OmegaConf.to_object(hydra_cfg.overrides.task))
    return OmegaConf.merge(prev_cfg, overrides)


def check_cfg(cfg, log):

    # exit
    if (cfg.prev_exp_dir and cfg.train) and not cfg.training.warm_start:
        log.error(
            "Rerunning experiment with train=True but warm_start=False. Exiting to avoid overwrite."
        )
        sys.exit()
