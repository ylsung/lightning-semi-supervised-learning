import os
import torch
import collections
import numpy as np
import torch.nn as nn
from time import time
from copy import deepcopy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from lightning_ssl.utils.torch_utils import (
    half_mixup_data,
    soft_cross_entropy,
    l2_distribution_loss,
    smooth_label,
    customized_weight_decay,
    interleave,
    sharpening,
)

from lightning_ssl.module.classifier_module import ClassifierModule
from lightning_ssl.module.mixmatch import Mixmatch


class MixmatchModule(ClassifierModule, Mixmatch):
    def __init__(self, hparams, classifier, loaders):
        super(MixmatchModule, self).__init__(hparams, classifier, loaders)
        self.lambda_u = 0
        self.rampup_length = 16384  # get from the mixmatch codes

        for key, size in zip(["lab_loss", "unl_loss"], [500, 500]):
            self.train_dict[key] = collections.deque([], size)

    def on_train_batch_end(self, *args, **kwargs):
        # self.wdm.decay()
        self.ema.step()

        # linearly ramp up lambda_u
        if self.lambda_u == self.hparams.lambda_u:
            return

        step = (self.hparams.lambda_u - 0) / self.rampup_length
        self.lambda_u = min(self.lambda_u + step, self.hparams.lambda_u)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        labeled_batch, unlabeled_batch = batch[0], batch[1:]
        # labeled_batch, unlabeled_batch = batch
        labeled_x, labeled_y = labeled_batch

        labeled_y = smooth_label(
            labeled_y, self.hparams.n_classes, self.hparams.label_smoothing
        )

        unlabeled_xs = [b[0] for b in unlabeled_batch]  # only get the images

        loss, l_l, l_u = self._loss(
            labeled_x,
            labeled_y,
            unlabeled_xs,
            batch_inference=self.hparams.batch_inference,
        )

        loss = l_l + self.lambda_u * l_u

        # y = torch.argmax(mixed_target, dim=-1)
        # acc = self.accuracy(logits[:batch_size], y[:batch_size])

        self.train_dict["loss"].append(loss.item())
        # self.train_dict["acc"].append(acc.item())
        self.train_dict["lab_loss"].append(l_l.item())
        self.train_dict["unl_loss"].append(l_u.item())

        # tensorboard_logs = {"train/loss": np.mean(self.train_dict["loss"]),
        #                     # "train/acc": np.mean(self.train_dict["acc"]),
        #                     "train/lab_loss": np.mean(self.train_dict["lab_loss"]),
        #                     "train/unl_loss": np.mean(self.train_dict["unl_loss"]),
        #                     "lambda_u": self.lambda_u}

        # progress_bar = {# "acc": np.mean(self.train_dict["acc"]),
        #                 "lab_loss": np.mean(self.train_dict["lab_loss"]),
        #                 "unl_loss": np.mean(self.train_dict["unl_loss"]),
        #                 "lambda_u": self.lambda_u}

        # return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar}

        self.log(
            "train/loss", np.mean(self.train_dict["loss"]), prog_bar=False, logger=True
        )
        self.log(
            "train/lab_loss",
            np.mean(self.train_dict["lab_loss"]),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/unl_loss",
            np.mean(self.train_dict["unl_loss"]),
            prog_bar=True,
            logger=True,
        )
        self.log("train/lambda_u", self.lambda_u, prog_bar=False, logger=True)

        return {"loss": loss}

    # def validation_epoch_end(self, outputs):
    #     # Monitor both validation set and test set
    #     # record the test accuracies of last 20 checkpoints

    #     avg_loss_list, avg_acc_list = [], []

    #     for output in outputs:
    #         avg_loss = torch.stack([x["val_loss"] * x["val_num"] for x in output]).sum() / \
    #             np.sum([x["val_num"] for x in output])
    #         avg_acc = torch.stack([x["val_acc"] * x["val_num"] for x in output]).sum() / \
    #             np.sum([x["val_num"] for x in output])

    #         avg_loss_list.append(avg_loss)
    #         avg_acc_list.append(avg_acc)

    #     # record best results of validation set
    #     self.train_dict["val_acc"][0] = max(self.train_dict["val_acc"][0], avg_acc_list[0].item())
    #     self.train_dict["test_acc"].append(avg_acc_list[1].item())

    #     # tensorboard_logs = {"val/loss": avg_loss_list[0],
    #     #                     "val/acc": avg_acc_list[0],
    #     #                     "val/best_acc": self.train_dict["val_acc"][0],
    #     #                     "test/median_acc": np.median(self.train_dict["test_acc"])}

    #     # return {"val_loss": avg_loss_list[0], "val_acc": avg_acc_list[0], "log": tensorboard_logs}

    #     self.log("val/loss", avg_loss_list[0], on_step=True, on_epoch=False,
    #         prog_bar=False, logger=True)
    #     self.log("val/acc", avg_acc_list[0], on_step=True, on_epoch=False,
    #         prog_bar=False, logger=True)
    #     self.log("val/best_acc", self.train_dict["val_acc"][0], on_step=True, on_epoch=False,
    #         prog_bar=False, logger=True)
    #     self.log("test/median_acc", np.median(self.train_dict["test_acc"]), on_step=True, on_epoch=False,
    #         prog_bar=False, logger=True)
