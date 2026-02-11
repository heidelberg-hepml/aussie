import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict

log = logging.getLogger("Trainer")

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        cfg: DictConfig,
        exp_dir: str,
        device: torch.device,
        use_amp=False,
    ):
        """
        model           -- a pytorch model to be trained
        dataloaders     -- a dictionary containing pytorch data loaders at keys 'train' and 'val'
        cfg             -- configuration dictionary
        exp_dir         -- directory to which training outputs will be saved
        """

        self.model = model
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = device
        self.use_amp = use_amp

        self.start_epoch = 0
        self.patience_counter = 0

    def prepare_training(self):

        log.info("Preparing model training")

        # init optimizer
        self.optimizer = instantiate(
            self.cfg.optimizer, params=self.model.trainable_parameters, lr=self.cfg.lr
        )

        # init scaler
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # init scheduler
        self.steps_per_epoch = len(self.dataloaders["train"])
        if self.cfg.scheduler:
            self.scheduler = self.init_scheduler()

        # set logging of metrics
        if self.cfg.use_tensorboard:
            self.summarizer = SummaryWriter(self.exp_dir)
            log.info(f"Writing tensorboard summaries to dir {self.exp_dir}")
        else:
            log.info("`use_tensorboard` set to False. No summaries will be written")

        self.epoch_train_losses = np.array([])
        self.epoch_val_losses = np.array([])
        self.best_val_loss = np.inf

        if self.cfg.warm_start:
            checkpoint = f"model{self.cfg.warm_start_epoch or ''}.pt"
            path = os.path.join(self.exp_dir, checkpoint)
            self.load(path)
            # avoid overriding checkpoint
            os.rename(path, path.replace(".pt", f"_old.pt"))
            log.info(f"Warm starting training from epoch {self.start_epoch}")

        # compile model
        self.model = torch.compile(self.model)

    def run_training(self):

        self.prepare_training()

        num_epochs = self.cfg.epochs - self.start_epoch
        log.info(f"Beginning training loop with epochs set to {num_epochs}")
        if self.cfg.patience:
            log.info(f"Early stopping patience set to {self.cfg.patience}")

        t0_total = time.time()
        for e in range(num_epochs):

            self.epoch = (self.start_epoch or 0) + e

            # train
            self.train_one_epoch()

            # validate at given frequency
            if (self.epoch + 1) % self.cfg.validate_freq == 0:

                self.validate_one_epoch()

                # check whether validation loss improved
                if (val_loss := self.epoch_val_losses[-1]) < self.best_val_loss:
                    self.patience_counter = 0

                    if self.cfg.save_best_epoch:  # save best checkpoint
                        self.best_val_loss = val_loss
                        self.save()

                elif self.cfg.patience:  # early stopping
                    self.patience_counter += 1
                    if self.patience_counter == self.cfg.patience:
                        log.info(f"Stopping training early at epoch {self.epoch}")
                        break

            # optionally save model at given frequency
            if save_freq := self.cfg.save_freq:
                if (self.epoch + 1) % save_freq == 0 or self.epoch == 0:
                    self.save(tag=self.epoch)

            # estimate training time
            if e == 0:
                t0_epoch = time.time()
            if e == 1:
                dtEst = (time.time() - t0_epoch) * num_epochs
                log.info(
                    f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h"
                )

        traintime = time.time() - t0_total
        log.info(
            f"Finished training {self.epoch + 1} epochs after {traintime:.2f} s"
            f" = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h."
        )

        # save final model
        if not self.cfg.save_best_epoch:
            log.info("Saving final model")
            self.save()

    def train_one_epoch(self):

        # set modules to training mode
        self.model.train()  # NOTE: Ensure frozen submodules are set to eval mode each batch!
        try:
            self.optimizer.train()
        except AttributeError:
            pass

        # create list to save loss per iteration
        train_losses = []

        # iterate batch wise over input
        for itr, batch in enumerate(self.dataloaders["train"]):

            batch = batch.to(self.device, non_blocking=True)

            # calculate batch loss
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.model.batch_loss(batch)

            # update model parameters
            step = itr + self.epoch * self.steps_per_epoch
            total_steps = self.cfg.epochs * self.steps_per_epoch
            self.model.update(loss, self.optimizer, self.scaler, step, total_steps)

            # update learning rate
            if self.cfg.scheduler:
                self.scheduler.step()

            # track loss
            train_losses.append(loss.detach())
            if self.cfg.use_tensorboard and (not step % self.cfg.log_iters) or not step:
                iter_loss = torch.stack(train_losses[-self.cfg.log_iters :])
                self.summarizer.add_scalar(
                    "iter_loss_train",
                    iter_loss.mean().cpu().numpy(),
                    step,
                )
                for k, v in self.model.log_buffer.items():  # model scalars
                    self.summarizer.add_scalar(
                        k,
                        torch.stack(v).mean().cpu().numpy(),
                        step,
                    )
                self.model.log_buffer.clear()

        # track loss
        self.epoch_train_losses = np.append(
            self.epoch_train_losses, torch.stack(train_losses).mean().cpu().numpy()
        )

        # optionally log to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_train", self.epoch_train_losses[-1], self.epoch
            )
            if self.cfg.scheduler:
                self.summarizer.add_scalar(
                    "learning_rate", self.scheduler.get_last_lr()[0], self.epoch
                )

    @torch.no_grad()
    def validate_one_epoch(self):

        # set modules to evaluation mode
        self.model.eval()
        try:
            self.optimizer.eval()
        except AttributeError:
            pass

        # calculate loss batchwise over input
        val_losses = []
        for batch in self.dataloaders["val"]:

            batch = batch.to(self.device, non_blocking=True)
            # calculate loss
            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.model.batch_loss(batch)
            val_losses.append(loss.detach())

        # track loss
        self.epoch_val_losses = np.append(
            self.epoch_val_losses, torch.stack(val_losses).mean().cpu().numpy()
        )

        # optional logging to tensorboard
        if self.cfg.use_tensorboard:
            self.summarizer.add_scalar(
                "epoch_loss_val", self.epoch_val_losses[-1], self.epoch
            )

    def save(self, tag=""):
        """Save the model along with the training state"""

        # set modules to evaluation mode
        self.model.eval()
        try:
            self.optimizer.eval()
        except AttributeError:
            pass

        model_dict = {
            k.replace("_orig_mod.", ""): v for k, v in self.model.state_dict().items()
        }
        state_dicts = {
            "opt": self.optimizer.state_dict(),
            "model": model_dict,
            "train_losses": self.epoch_train_losses,
            "val_losses": self.epoch_val_losses,
            "epoch": self.epoch,
        }
        if self.cfg.scheduler:
            state_dicts["scheduler"] = self.scheduler.state_dict()
        torch.save(state_dicts, os.path.join(self.exp_dir, f"model{tag}.pt"))

    def load(self, path):
        """Load the model and training state"""

        state_dicts = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dicts["model"])
        if "train_losses" in state_dicts:
            self.epoch_train_losses = state_dicts.get("train_losses", {})
        if "val_losses" in state_dicts:
            self.epoch_val_losses = state_dicts.get("val_losses", {})
            if len(self.epoch_val_losses) > 0:
                self.best_val_loss = self.epoch_val_losses.min()
        if "epoch" in state_dicts:
            self.start_epoch = state_dicts.get("epoch", 0) + 1
        if "opt" in state_dicts:
            self.optimizer.load_state_dict(state_dicts["opt"])
        if "scheduler" in state_dicts:
            self.scheduler.load_state_dict(state_dicts["scheduler"])
        self.model.net.to(self.device)

    def init_scheduler(self):
        scfg = self.cfg.scheduler
        name = scfg._target_
        total_steps = self.cfg.epochs * self.steps_per_epoch
        match name:
            case "torch.optim.lr_scheduler.OneCycleLR":
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    total_steps=total_steps,
                )
            case "torch.optim.lr_scheduler.StepLR":
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    step_size=self.steps_per_epoch * scfg.step_size,
                )
            case _:
                return instantiate(
                    scfg,
                    optimizer=self.optimizer,
                    # total_iters=total_steps,
                )
