import torch
import os
import numpy as np
from abc import abstractmethod
from hydra.utils import call, instantiate
from torch.utils.data import DataLoader, random_split

from src.experiments.base_experiment import BaseExperiment
from src.datasets import UnfoldingData
from src.utils.trainer import Trainer


class TrainingExperiment(BaseExperiment):

    def run(self):

        # preprocessing and observables
        self.process = instantiate(self.cfg.dataset.process)

        if self.cfg.train:

            # model
            if not hasattr(self, "model"):
                self.init_model()

            # initialize dataloaders in train mode
            self.log.info("Creating dataLoaders")
            self.dataloaders = dict(
                zip(("train", "val", "test"), self.init_dataloader(training=True))
            )

        if self.cfg.train:

            # train model
            self.log.info("Running training")

            trainer = Trainer(
                model=self.model,
                dataloaders=self.dataloaders,
                cfg=self.cfg.training,
                exp_dir=self.exp_dir,
                device=self.device,
                use_amp=self.cfg.use_amp,
            )
            trainer.run_training()

            del self.dataloaders
            torch.cuda.empty_cache()

        # evaluate model
        if self.cfg.evaluate:

            # model
            if not hasattr(self, "model"):
                self.init_model()

            # initialize dataloaders (without drop_last)
            self.log.info("Creating dataLoaders")
            self.dataloaders = dict(
                zip(("train", "val", "test"), self.init_dataloader(training=False))
            )

            # load model state
            self.log.info(f"Loading model state from {self.exp_dir}.")
            self.model.load(self.exp_dir, self.device)
            self.model.eval()

            self.log.info("Running evaluation on test dataset")
            for k, d in self.dataloaders.items():
                predictions = self.evaluate(d)
                # save to disk
                savepath = os.path.join(self.exp_dir, f"predictions_{k}.npz")
                self.log.info(f"Saving {k} predictions to {savepath}")
                np.savez(savepath, **predictions)

            del self.dataloaders
            torch.cuda.empty_cache()

        # make plots
        if self.cfg.plot:
            self.log.info("Making plots")
            self.plot()

        # print memory usage
        self.log_resources()

    def init_model(self):
        self.log.info("Initializing model")

        self.model = call(self.cfg.model, self.cfg)
        self.model = self.model.to(self.device)
        model_name = (
            f"{self.model.__class__.__name__}[{self.model.net.__class__.__name__}]"
        )
        num_params = sum(w.numel() for w in self.model.trainable_parameters)
        self.log.info(f"Model ({model_name}) has {num_params} trainable parameters")

    def init_dataloader(self, training=False):

        dcfg = self.cfg.data
        tcfg = self.cfg.training

        # read data
        dset = instantiate(self.cfg.dataset.reader)

        # preprocess (on cpu)
        for transform in self.process.transforms:
            dset = transform.forward(dset)

        # optionally move dataset to gpu
        on_gpu = dcfg.on_gpu and self.cfg.use_gpu
        if on_gpu:
            dset = dset.to(self.device)

        # split dataset
        dsets = self.split_dataset(dset)

        # load sim weights
        if (p := self.cfg.prev_it_path) is not None:

            # create unit weights if tensor is absent
            if dset.sample_logweights is None:
                dset.sample_logweights = torch.zeros(
                    len(dset), dtype=torch.float, device=dset.device
                )

            for i, k in enumerate(("train", "val", "test")):

                idcs = torch.as_tensor(dsets[i].indices, device=dset.device)

                sim_logweights = torch.from_numpy(
                    np.load(os.path.join(p, f"unf/predictions_{k}.npz"))[
                        "lw_z_sim"
                    ].mean(0)
                ).to(dset.device)

                if dset.labels is None:
                    # all events are sim if labels absent
                    dset.sample_logweights[idcs] = sim_logweights
                else:
                    # fill sim weights only
                    mask = dset.labels[idcs] == 0
                    dset.sample_logweights.index_put_((idcs[mask],), sim_logweights)

        self.log.info(f"Read dataset:\n{dset}")

        # create dataloaders
        dataloaders = []
        num_workers = (
            0 if on_gpu or not self.cfg.train else max(self.cfg.num_cpus - 1, 0)
        )
        use_mp = self.cfg.train and num_workers > 0
        for i, d in enumerate(dsets):

            is_train_split = (i == 0) and training
            batch_size = tcfg.batch_size if is_train_split else tcfg.test_batch_size

            dataloaders.append(
                DataLoader(
                    d,
                    shuffle=is_train_split,
                    drop_last=is_train_split,
                    batch_size=batch_size,
                    collate_fn=self.collate_fn,
                    num_workers=num_workers,
                    pin_memory=not on_gpu,
                    multiprocessing_context="spawn" if use_mp else None,
                    persistent_workers=use_mp,
                )
            )

        return dataloaders

    def split_dataset(self, dset):

        dcfg = self.cfg.data

        # create splits
        assert dcfg.val_frac > 0, "A validation split is required"
        assert dcfg.test_frac > 0, "A testing split is required"

        # seed data split to avoid leakage across iterations
        fixed_rng = torch.Generator().manual_seed(1729)
        splits = random_split(
            dset,
            [1 - dcfg.val_frac - dcfg.test_frac, dcfg.val_frac, dcfg.test_frac],
            generator=fixed_rng,
        )

        return list(splits)

    def collate_fn(self, batch: UnfoldingData):
        "Perform experiment-specific collation. Can help to avoid CPU-GPU sync during training."
        return batch

    @abstractmethod
    def init_dataset(self, path_exp, path_sim):
        "Read and return a dataset. To be implemented by the child class"
        pass

    @abstractmethod
    def evaluate(self):
        "Iterate dataset and save model predictions. To be implemented by the child class"
        pass

    @abstractmethod
    def plot(self):
        "Create and save evaluation plots. To be implemented by the child class"
        pass
