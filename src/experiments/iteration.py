import os

from omegaconf import OmegaConf

from src.experiments import ClassificationExperiment, UnfoldingExperiment
from src.experiments.base_experiment import BaseExperiment

class IterationExperiment(BaseExperiment):

    def run(self):

        # dispatch settings to steps 1 and 2
        self.resolve_configs()

        # iterate
        if self.cfg.prev_it_path is None:
            start_it = 1
        else:
            start_it = int(self.cfg.prev_it_path.split("it_")[-1].strip("/")) + 1

        for it in range(start_it, start_it + self.cfg.iterations):

            self.log.info(f"Starting iteration {it}")

            # point to previous iteration if it exists
            self.prev_it_path = os.path.dirname(unf_dir) if it > 1 else self.cfg.prev_it_path

            # run step one
            if (self.cfg.cls_path is None) or it > 1:
                cls_dir = self.run_cls(it)
            else:
                # skip classification if cls_path provided
                # TODO: Assert consistency of cls_path and prev_it_path
                self.log.info(f"Skipping classification step")
                cls_dir = self.cfg.cls_path

            # run step two
            unf_dir = self.run_unf(it, cls_dir)

    def run_cls(self, it):

        cfg = self.cfg.cls

        # create experiment directory
        cls_dir = os.path.join(self.exp_dir, f"it_{it}/cls")
        os.makedirs(cls_dir + "/.hydra")

        # link previous iteration
        cfg.prev_it_path = self.prev_it_path

        # optionally decay learning rate
        if (decay := self.cfg.cls_lr_decay) is not None:
            cfg.training.lr *= decay ** (it - 1)

        # write config
        with open(cls_dir + "/.hydra/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # initialize and run
        exp = ClassificationExperiment(cfg, cls_dir)
        exp.run()

        return cls_dir

    def run_unf(self, it, cls_dir):

        cfg = self.cfg.unf

        # create experiment directory
        unf_dir = os.path.join(self.exp_dir, f"it_{it}/unf")
        os.makedirs(unf_dir + "/.hydra")

        # link step one and previous iteration
        cfg.cls_path = cls_dir
        cfg.prev_it_path = self.prev_it_path

        # optionally decay learning rate
        if (decay := self.cfg.unf_lr_decay) is not None:
            cfg.training.lr *= decay ** (it - 1)

        # write config
        with open(unf_dir + "/.hydra/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        # initialize
        exp = UnfoldingExperiment(cfg, unf_dir)
        # optionally warm start
        if self.cfg.warm_start_unf and self.prev_it_path:
            assert cfg.iterate_target
            self.log.info(f"Warm starting unf net from iteration {it-1}")
            exp.init_model()
            exp.model.load_nets(self.prev_it_path, self.device)

        # run
        exp.run()

        return unf_dir

    def resolve_configs(self):

        # dataset
        self.cfg.cls.dataset = self.cfg.dataset
        self.cfg.unf.dataset = self.cfg.dataset

        # gpu
        self.cfg.cls.use_gpu = self.cfg.use_gpu
        self.cfg.unf.use_gpu = self.cfg.use_gpu

        # f32 precision
        self.cfg.cls.use_tf32 = self.cfg.use_tf32
        self.cfg.unf.use_tf32 = self.cfg.use_tf32

        # mixed precision
        self.cfg.cls.use_amp = self.cfg.use_amp
        self.cfg.unf.use_amp = self.cfg.use_amp