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
from lightning_ssl.utils.torch_utils import EMA, LabelSmoothingLoss, smooth_label, soft_cross_entropy, \
    mixup_data, customized_weight_decay

# Use the style similar to pytorch_lightning (pl)
# Codes will revised to be compatible with pl when pl has all the necessary features.

# Codes borrowed from 
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=x-34xKCI40yW

ema_classifier = None

class NewClassifierModule(pl.LightningModule):

    def __init__(self, hparams, classifier, loaders):
        super(NewClassifierModule, self).__init__()
        self.hparams = hparams
        self.classifier = classifier
        global ema_classifier
        ema_classifier = deepcopy(classifier)
        ema_classifier.cuda()
        ema_classifier.eval()
        self.loss_module = LabelSmoothingLoss(hparams.n_classes, hparams.label_smoothing)
        self.loaders = loaders
        self.best_dict = {
            "val_acc": 0,
        }
        self.max_size = 500

        self.train_dict = {key: collections.deque([], self.max_size) for key in ["loss", "acc"]}
    
    def forward(self, x, model):
        return model(x)

    def on_train_start(self):
        global ema_classifier
        # self.ema = EMA(self.hparams.ema, self.ema_classifier, wd)
        self.ema = EMA(self.classifier, ema_classifier, self.hparams.ema)
        # self.ema = EMA(self.hparams.ema, self.classifier)
    
    def on_batch_end(self):
        # self.ema.update(self.classifier)
        # wd = self.hparams.weight_decay * self.hparams.learning_rate
        customized_weight_decay(self.classifier, self.hparams.weight_decay)
        self.ema.step()

    def accuracy(self, y_hat, y):
        return 100 * (torch.argmax(y_hat, dim=-1) == y).float().mean()
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        # return smooth one-hot like label
        y_one_hot = smooth_label(y, self.hparams.n_classes, self.hparams.label_smoothing)
        # mixup
        mixed_x, mixed_y = mixup_data(x, y_one_hot, self.hparams.alpha)
        y_hat = self.forward(mixed_x, self.classifier)
        loss = soft_cross_entropy(y_hat, mixed_y)
        y = torch.argmax(mixed_y, dim=-1)
        acc = self.accuracy(y_hat, y)
        num = len(y)

        self.train_dict["loss"].append(loss.item())
        self.train_dict["acc"].append(acc.item())

        tensorboard_logs = {"train/loss": np.mean(self.train_dict["loss"]), 
                            "train/acc": np.mean(self.train_dict["acc"])}

        progress_bar = {"acc": np.mean(self.train_dict["acc"])}

        return {"loss": loss, "train_acc": acc, 
            "train_num": num, "log": tensorboard_logs, 
            "progress_bar": progress_bar}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        global ema_classifier
        x, y = batch
        y_hat = self.forward(x, ema_classifier)

        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {"val_loss": F.cross_entropy(y_hat, y), 
                "val_acc": acc,
                "val_num": num}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] * x["val_num"] for x in outputs]).sum() / \
            np.sum([x["val_num"] for x in outputs])
        avg_acc = torch.stack([x["val_acc"] * x["val_num"] for x in outputs]).sum() / \
            np.sum([x["val_num"] for x in outputs])

        self.best_dict["val_acc"] = max(self.best_dict["val_acc"], avg_acc.item())

        tensorboard_logs = {"val/loss": avg_loss, "val/acc": avg_acc, 
            "val/best_acc": self.best_dict["val_acc"]}

        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        global ema_classifier
        x, y = batch
        y_hat = self.forward(x, ema_classifier)
        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {"test_loss": F.cross_entropy(y_hat, y),
                "test_acc": acc,
                "test_num": num}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] * x["test_num"] for x in outputs]).sum() / \
            np.sum([x["test_num"] for x in outputs])
        avg_acc = torch.stack([x["test_acc"] * x["test_num"] for x in outputs]).sum() / \
            np.sum([x["test_num"] for x in outputs])
        logs = {"test/loss": avg_loss, "test/acc": avg_acc}
        return {"test_loss": avg_loss, "test_acc": avg_acc, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        opt = torch.optim.Adam(self.classifier.parameters(), lr=self.hparams.learning_rate)
        # opt = torch.optim.SGD(self.classifier.parameters(), self.hparams.learning_rate,
        #                       momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, float(self.hparams.max_epoch))
        return [opt] # , [scheduler]

    def train_dataloader(self):
        # REQUIRED
        return self.loaders["tr_loader"]

    def val_dataloader(self):
        # OPTIONAL
        return self.loaders["va_loader"]

    def test_dataloader(self):
        # OPTIONAL
        return self.loaders["te_loader"]