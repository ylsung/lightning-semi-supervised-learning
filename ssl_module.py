import os
import torch
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from algorithms import EMA, LabelSmoothingLoss, smooth_label, soft_cross_entropy, \
    mixup_data, customized_weight_decay

# Use the style similar to pytorch_lightning (pl)
# Codes will revised to be compatible with pl when pl has all the necessary features.

# Codes borrowed from 
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=x-34xKCI40yW

class ClassifierModule(pl.LightningModule):

    def __init__(self, hparams, classifier, loaders):
        super(ClassifierModule, self).__init__()
        self.hparams = hparams
        self.classifier = classifier # the relu and batchnorm of model should update
        self.ema_classifier = deepcopy(classifier)
        self.loss_module = LabelSmoothingLoss(hparams.n_classes, hparams.label_smoothing)
        self.loaders = loaders
        self.best_dict = {
            "val_acc": 0,
        }
        self.use_ema = True if self.hparams.ema != 0 else False
        
    def forward(self, x, model):
        return model(x)

    def on_train_start(self):
        # self.ema = EMA(self.hparams.ema, self.ema_classifier, wd)
        self.ema = EMA(self.classifier, self.ema_classifier, self.hparams.ema)
        # self.ema = EMA(self.hparams.ema, self.classifier)
    
    def on_batch_end(self):
        # self.ema.update(self.classifier)
        # wd = self.hparams.weight_decay * self.hparams.learning_rate
        customized_weight_decay(self.classifier, self.hparams.weight_decay)
        self.ema.step()

    def accuracy(self, y_hat, y):
        return 100 * (torch.argmax(y_hat, dim=-1) == y).float().mean()
    
    def supervised_train_step(self, batch, batch_nb):
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
        tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        return {"loss": loss, "train_acc": acc, "train_num": num, "log": tensorboard_logs}

    def semi_train_step(self, batch, batch_nb):
        label_batch, unlabel_batch = batch[0], batch[1]
        pass

    def training_step(self, batch, batch_nb):
        # REQUIRED
        if self.hparams.learning_scenario == "supervised":
            return self.supervised_train_step(batch, batch_nb)
        elif self.hparams.learning_scenario == "semi":
            pass

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x, self.ema_classifier)

        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {"val_loss": F.cross_entropy(y_hat, y), 
                "val_acc": acc,
                "val_num": num}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] * x["val_num"] for x in outputs]).sum() / \
            np.sum([x["val_num"] for x in outputs])

        self.best_dict["val_acc"] = max(self.best_dict["val_acc"], avg_acc)

        tensorboard_logs = {"avg_val_loss": avg_loss, "avg_val_acc": avg_acc, 
            "best_val_acc": self.best_dict["val_acc"]}

        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x, self.ema_classifier)
        acc = self.accuracy(y_hat, y)
        num = len(y)

        return {"test_loss": F.cross_entropy(y_hat, y),
                "test_acc": acc,
                "test_num": num}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] * x["test_num"] for x in outputs]).sum() / \
            np.sum([x["test_num"] for x in outputs])
        logs = {"avg_test_loss": avg_loss, "avg_test_acc": avg_acc}
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