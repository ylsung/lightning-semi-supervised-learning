import os
import torch
import collections
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from lightning_ssl.utils.torch_utils import (
    EMA,
    smooth_label,
    soft_cross_entropy,
    mixup_data,
    customized_weight_decay,
    WeightDecayModule,
    split_weight_decay_weights,
)

# Use the style similar to pytorch_lightning (pl)
# Codes will revised to be compatible with pl when pl has all the necessary features.

# Codes borrowed from
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=x-34xKCI40yW


class ClassifierModule(pl.LightningModule):
    def __init__(self, hparams, classifier, loaders):
        super(ClassifierModule, self).__init__()
        self.hparams = hparams
        self.classifier = classifier
        self.ema_classifier = deepcopy(classifier)
        self.loaders = loaders
        self.best_dict = {
            "val_acc": 0,
        }
        self.train_dict = {
            key: collections.deque([], size)
            for key, size in zip(
                ["loss", "acc", "val_acc", "test_acc"], [500, 500, 1, 20]
            )
        }

        # to record the best validation accuracy
        self.train_dict["val_acc"].append(0)  # avoid empty list

    def on_train_start(self):
        # model will put in GPU before this function
        # so we initiate EMA and WeightDecayModule here
        self.ema = EMA(self.classifier, self.ema_classifier, self.hparams.ema)
        # self.wdm = WeightDecayModule(self.classifier, self.hparams.weight_decay, ["bn", "bias"])

    def on_train_batch_end(self, *args, **kwargs):
        # self.ema.update(self.classifier)
        # wd = self.hparams.weight_decay * self.hparams.learning_rate
        # customized_weight_decay(self.classifier, self.hparams.weight_decay, ["bn", "bias"])
        # self.wdm.decay()
        self.ema.step()

    def accuracy(self, y_hat, y):
        return 100 * (torch.argmax(y_hat, dim=-1) == y).float().mean()

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        # return smooth one-hot like label
        y_one_hot = smooth_label(
            y, self.hparams.n_classes, self.hparams.label_smoothing
        )
        # mixup
        mixed_x, mixed_y = mixup_data(x, y_one_hot, self.hparams.alpha)
        y_hat = self.classifier(mixed_x)
        loss = soft_cross_entropy(y_hat, mixed_y)
        y = torch.argmax(mixed_y, dim=-1)
        acc = self.accuracy(y_hat, y)
        num = len(y)

        self.train_dict["loss"].append(loss.item())
        self.train_dict["acc"].append(acc.item())

        # tensorboard_logs = {"train/loss": np.mean(self.train_dict["loss"]),
        #                     "train/acc": np.mean(self.train_dict["acc"])}

        # progress_bar = {"acc": np.mean(self.train_dict["acc"])}

        # return {"loss": loss, "train_acc": acc,
        #     "train_num": num, "log": tensorboard_logs,
        #     "progress_bar": progress_bar}

        self.log(
            "train/loss", np.mean(self.train_dict["loss"]), prog_bar=False, logger=True
        )
        self.log(
            "train/acc", np.mean(self.train_dict["acc"]), prog_bar=False, logger=True
        )

        return {"loss": loss, "train_acc": acc, "train_num": num}

    def validation_step(self, batch, *args):
        # OPTIONAL
        x, y = batch
        y_hat = self.ema_classifier(x)

        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {"val_loss": F.cross_entropy(y_hat, y), "val_acc": acc, "val_num": num}

    def validation_epoch_end(self, outputs):
        # Monitor both validation set and test set
        # record the test accuracies of last 20 checkpoints

        avg_loss_list, avg_acc_list = [], []

        for output in outputs:
            avg_loss = torch.stack(
                [x["val_loss"] * x["val_num"] for x in output]
            ).sum() / np.sum([x["val_num"] for x in output])
            avg_acc = torch.stack(
                [x["val_acc"] * x["val_num"] for x in output]
            ).sum() / np.sum([x["val_num"] for x in output])

            avg_loss_list.append(avg_loss)
            avg_acc_list.append(avg_acc)

        # record best results of validation set
        self.train_dict["val_acc"][0] = max(
            self.train_dict["val_acc"][0], avg_acc_list[0].item()
        )
        self.train_dict["test_acc"].append(avg_acc_list[1].item())

        # tensorboard_logs = {"val/loss": avg_loss_list[0],
        #                     "val/acc": avg_acc_list[0],
        #                     "val/best_acc": self.train_dict["val_acc"][0],
        #                     "test/median_acc": np.median(self.train_dict["test_acc"])}

        # return {"val_loss": avg_loss_list[0], "val_acc": avg_acc_list[0], "log": tensorboard_logs}
        self.logger.experiment.add_scalar(
            "t", np.median(self.train_dict["test_acc"]), self.global_step
        )
        self.log("val/loss", avg_loss_list[0], prog_bar=False, logger=True)
        self.log("val/acc", avg_acc_list[0], prog_bar=False, logger=True)
        self.log(
            "val/best_acc", self.train_dict["val_acc"][0], prog_bar=False, logger=True
        )
        self.log(
            "test/median_acc",
            np.median(self.train_dict["test_acc"]),
            prog_bar=False,
            logger=True,
        )

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.ema_classifier(x)
        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {
            "test_loss": F.cross_entropy(y_hat, y),
            "test_acc": acc,
            "test_num": num,
        }

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack(
            [x["test_loss"] * x["test_num"] for x in outputs]
        ).sum() / np.sum([x["test_num"] for x in outputs])
        avg_acc = torch.stack(
            [x["test_acc"] * x["test_num"] for x in outputs]
        ).sum() / np.sum([x["test_num"] for x in outputs])
        # logs = {"test/loss": avg_loss, "test/acc": avg_acc}
        # return {"test_loss": avg_loss, "test_acc": avg_acc, "log": logs, "progress_bar": logs}

        self.log("test/loss", avg_loss, prog_bar=False, logger=True)
        self.log("test/acc", avg_acc, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)

        # split the weights into need weight decay and no need weight decay
        parameters = split_weight_decay_weights(
            self.classifier, self.hparams.weight_decay, ["bn", "bias"]
        )

        opt = torch.optim.AdamW(
            parameters, lr=self.hparams.learning_rate, weight_decay=0
        )
        # opt = torch.optim.SGD(self.classifier.parameters(), self.hparams.learning_rate,
        #                       momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(self.hparams.max_epoch))
        return [opt]  # , [scheduler]

    # def train_dataloader(self):
    #     # REQUIRED
    #     return self.loaders["tr_loader"]

    # def val_dataloader(self):
    #     # OPTIONAL
    #     return [self.loaders["va_loader"], self.loaders["te_loader"]]

    # def test_dataloader(self):
    #     # OPTIONAL
    #     return self.loaders["te_loader"]
