import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from collections import defaultdict
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages

from src.experiments.training import TrainingExperiment
from src.utils import plotting


class UnfoldingExperiment(TrainingExperiment):

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the model on the test dataset.
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
            lw_z_sim = []
            for batch in dataloader:

                z = batch.z[batch.labels == 0].to(self.device, non_blocking=True)
                lw_z_sim.append(self.model.net(z).squeeze(-1))

            predictions["lw_z_sim"].append(torch.cat(lw_z_sim).cpu())

        # stack
        predictions["lw_z_sim"] = torch.stack(predictions["lw_z_sim"]).numpy()

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

        # read (untransformed) test data
        self.log.info("Loading test data")
        dset = instantiate(self.cfg.dataset.reader)  # TODO: Just use test loader
        _, _, test_set = self.split_dataset(dset)

        # read predicted weights
        self.log.info("Reading predictions from disk")
        record = np.load(os.path.join(self.exp_dir, "predictions.npz"))

        mask_sim = test_set[:].labels == 0
        mask_dat = ~mask_sim
        lw_z_sim = record["lw_z_sim"].mean(0)

        # observables
        self.log.info("Plotting reco observables")
        x_dat = test_set[:].x[mask_dat]
        x_sim = test_set[:].x[mask_sim]
        with PdfPages(os.path.join(savedir, f"observables.pdf")) as pdf:
            for obs in self.process.observables:
                fig, ax = plotting.plot_reweighting(
                    exp=obs.compute(x_dat).numpy(),
                    sim=obs.compute(x_sim).numpy(),
                    weights_list=[np.exp(lw_z_sim)],
                    variance_list=[None],
                    names_list=["Unfolder"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel=obs.label,
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    logy=obs.logy,
                    qlims=obs.qlims,
                    xlims=obs.xlims,
                    # exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # latents
        self.log.info("Plotting part latents")
        z_dat = test_set[:].z[mask_dat]
        z_sim = test_set[:].z[mask_sim]
        with PdfPages(os.path.join(savedir, f"latents.pdf")) as pdf:
            for obs in self.process.observables:
                fig, ax = plotting.plot_reweighting(
                    exp=obs.compute(z_dat).numpy(),
                    sim=obs.compute(z_sim).numpy(),
                    weights_list=[np.exp(lw_z_sim)],
                    variance_list=[None],
                    names_list=["Unfolder"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel=obs.label,
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    logy=obs.logy,
                    qlims=obs.qlims,
                    xlims=obs.xlims,
                    # exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)
