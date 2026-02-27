import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from collections import defaultdict
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages

from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.experiments.training import TrainingExperiment
from src.utils import plotting


class ClassificationExperiment(TrainingExperiment):

    @torch.inference_mode()
    def evaluate(self, dataloader, tag=None):
        """
        Evaluates the Classifier on the test dataset.
        Predictions are saved alongside truth labels
        """

        self.model.eval()

        # get predictions across the test set
        predictions = defaultdict(list)

        # collect predictions
        lw_x = [
            self.model(batch.to(self.device, non_blocking=True)).squeeze(-1)
            for batch in dataloader
        ]

        if self.model.ensembled:
            predictions["lw_x"] = torch.cat(lw_x, dim=1).cpu()
        else:
            predictions["lw_x"] = torch.cat(lw_x, dim=0).unsqueeze(0).cpu()

        return predictions

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
        record = np.load(os.path.join(self.exp_dir, "predictions_test.npz"))

        labels = test_set[:].labels
        mask_sim = labels == 0
        mask_dat = ~mask_sim
        lw_x = record["lw_x"]
        lw_x_sim = lw_x[..., mask_sim.numpy()].mean(
            0
        )  # TODO: remove mean for ensemble uncertainties

        # load sample weights from iteration or data correction
        if (p := self.cfg.prev_it_path) is not None:

            lw_sample_sim = torch.from_numpy(
                np.load(os.path.join(p, "unf/predictions_test.npz"))["lw_z_sim"].mean(
                    0
                )
            )
            lw_x_sim += lw_sample_sim.numpy()

        if (lw_sample_exp := test_set[:].sample_logweights) is not None:  # exp weights
            exp_weights = lw_sample_exp[mask_dat].exp().numpy()
        else:
            exp_weights = None

        # latents
        self.log.info("Plotting part latents")
        if dset.aux_z is None:
            z_dat = test_set[:].z[mask_dat]
            z_sim = test_set[:].z[mask_sim]
        else:
            z_dat = test_set[:].aux_z[mask_dat]
            z_sim = test_set[:].aux_z[mask_sim]
        with PdfPages(os.path.join(savedir, "latents.pdf")) as pdf:
            for obs in self.process.observables_z:
                fig, ax = plotting.plot_reweighting(
                    exp=obs.compute(z_dat).numpy(),
                    sim=obs.compute(z_sim).numpy(),
                    weights_list=[np.exp(lw_x_sim)],
                    variance_list=[None],
                    names_list=["Classifier"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel=obs.label,
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    logy=obs.logy,
                    qlims=obs.qlims,
                    xlims=obs.xlims,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

        # observables
        self.log.info("Plotting reco observables")
        if dset.aux_x is None:
            x_dat = test_set[:].x[mask_dat]
            x_sim = test_set[:].x[mask_sim]
        else:
            x_dat = test_set[:].aux_x[mask_dat]
            x_sim = test_set[:].aux_x[mask_sim]
        with PdfPages(os.path.join(savedir, "observables.pdf")) as pdf:
            for obs in self.process.observables_x:
                fig, ax = plotting.plot_reweighting(
                    exp=obs.compute(x_dat).numpy(),
                    sim=obs.compute(x_sim).numpy(),
                    weights_list=[np.exp(lw_x_sim)],
                    variance_list=[None],
                    names_list=["Classifier"],
                    # xlabel=pcfg.obs_labels[i],
                    xlabel=obs.label,
                    figsize=np.array([1, 5 / 6]) * pw / 2,
                    num_bins=pcfg.num_bins,
                    discrete=obs.discrete,
                    logy=obs.logy,
                    qlims=obs.qlims,
                    xlims=obs.xlims,
                    exp_weights=exp_weights,
                )
                pdf.savefig(fig)
                plt.close(fig)

            labels = labels.int().numpy()
            wx = np.exp(lw_x_sim)
            lw_x = lw_x.mean(0)
            preds = expit(lw_x)
            sample_weights = np.ones(len(labels))
            # sample_weights[labels == 0] = wx # this would give the calibration AUC
            if exp_weights is not None:
                sample_weights[labels == 1] = exp_weights

            fig, ax = plotting.plot_reweighting(
                # fig, ax = plotting.plot_reweighting_ensemble(
                exp=lw_x[mask_dat],
                sim=lw_x[mask_sim],
                weights_list=[wx],
                variance_list=[None],
                names_list=["Classifier"],
                figsize=np.array([1, 5 / 6]) * pw / 2,
                num_bins=pcfg.num_bins,
                discrete=obs.discrete,
                exp_weights=exp_weights,
                xlabel=r"$\log R_\theta(x)$",
                logy=True,
                qlims=(1e-5, 1 - 1e-5),
                density=True,
                ratio_lims=(0.6, 1.4),
            )

            plt.subplots_adjust(top=0.9)
            auc = roc_auc_score(labels, preds, sample_weight=sample_weights)
            fig.suptitle(f"AUC = {auc:.5f}")

            pdf.savefig(fig)
            plt.close(fig)
