import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from collections import defaultdict
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils import plotting
from src.utils.datasets import StationData


class ClassificationExperiment(TrainingExperiment):

    # @property
    # def iterating(self):
    #     return self.cfg.w_prev_path is not None

    def init_preprocessing(self):

        self.train_keys = ["x_sim", "x_dat"]
        self.transforms = [instantiate(self.cfg.dataset.preprocessing).to(self.device)]

    @property
    def reader(self):

        # select dataset
        match name := self.cfg.dataset.name:

            case "omnifold":
                reader = StationData.read_omnifold
            case _:
                raise ValueError(f"Unknown dataset {name}")

        return reader

    def init_dataset(self):

        num = self.cfg.data.num or self.cfg.dataset.max_num
        dset = self.reader(
            path=self.cfg.dataset.dir,
            num=num,
            keys=self.train_keys,
        )

        return dset

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the Classifier on the test dataset.
        Predictions are saved alongside truth labels
        """

        self.model.eval()

        # get predictions across the test set
        predictions = defaultdict(list)
        n_evals = self.cfg.num_bnn_samples if self.model.bayesian else 1
        for _ in range(n_evals):

            if self.model.bayesian:  # sample new bnn weights
                self.model.reseed()

            # collect predictions
            lw_dat, lw_sim = [], []
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True)

                lw_dat.append(self.model(batch.x_dat).squeeze(-1))
                lw_sim.append(self.model(batch.x_sim).squeeze(-1))

            predictions["lw_x_dat"].append(torch.cat(lw_dat).cpu())
            predictions["lw_x_sim"].append(torch.cat(lw_sim).cpu())

        # stack
        predictions["lw_x_dat"] = torch.stack(predictions["lw_x_dat"])
        predictions["lw_x_sim"] = torch.stack(predictions["lw_x_sim"])

        # save to disk
        tag = "" if tag is None else f"_{tag}"
        savepath = os.path.join(self.exp_dir, f"predictions{tag}.npz")
        self.log.info(f"Saving {tag} labels, weights and probs to {savepath}")
        np.savez(savepath, **predictions)

    def plot(self):

        pcfg = self.cfg.plotting
        pw = pcfg.pagewidth

        savedir = os.path.join(self.exp_dir, "plots")
        os.makedirs(savedir, exist_ok=True)

        # load test data for histograms
        self.log.info("Loading test data")
        plot_keys = [
            "x_dat",
            "z_dat",
            "x_sim",
            "z_sim",
        ]

        num = self.cfg.data.num or self.cfg.dataset.max_num
        dset = self.reader(  # TODO: Just use test loader
            path=self.cfg.dataset.dir, keys=plot_keys, num=num
        )

        _, _, test_set = self.split_dataset(dset)

        # read predicted weights from disk
        self.log.info("Reading predictions from disk")
        record = np.load(os.path.join(self.exp_dir, "predictions.npz"))

        lw_dat, lw_sim = record["lw_x_dat"], record["lw_x_sim"]

        # observables
        self.log.info("Plotting reco observables")
        with PdfPages(os.path.join(savedir, f"observables.pdf")) as pdf:
            for i in range(self.cfg.dataset.dim):
                fig, ax = plotting.plot_reweighting(
                    exp=test_set[:].x_dat[:, i].numpy(),
                    sim=test_set[:].x_sim[:, i].numpy(),
                    weights_list=[np.exp(lw_sim)],
                    variance_list=[None],
                    names_list=["Classifier"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel="",
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=2 if i == 1 else False,
                    logy=True,
                    qlims=(5e-3, 1 - 5e-3),
                    # exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # latents
        self.log.info("Plotting part latents")
        with PdfPages(os.path.join(savedir, f"latents.pdf")) as pdf:
            for i in range(self.cfg.dataset.dim):
                fig, ax = plotting.plot_reweighting(
                    exp=test_set[:].z_dat[:, i].numpy(),
                    sim=test_set[:].z_sim[:, i].numpy(),
                    weights_list=[np.exp(lw_sim)],
                    variance_list=[None],
                    names_list=["Classifier"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel="",
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=2 if i == 1 else False,
                    logy=True,
                    qlims=(5e-3, 1 - 5e-3),
                    # exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)                
