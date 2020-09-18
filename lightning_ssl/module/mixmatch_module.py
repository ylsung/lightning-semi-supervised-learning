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
from .torch_utils import half_mixup_data, soft_cross_entropy, l2_distribution_loss, \
    smooth_label, customized_weight_decay, interleave

from .classifier_module import ClassifierModule 

# Use the style similar to pytorch_lightning (pl)
# Codes will revised to be compatible with pl when pl has all the necessary features.

# Codes borrowed from 
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=x-34xKCI40yW

def sharpening(label, T):
    label = label.pow(1 / T)
    return label / label.sum(-1, keepdim=True)

class MixmatchModule(ClassifierModule):

    def __init__(self, hparams, classifier, loaders):
        super(MixmatchModule, self).__init__(hparams, classifier, loaders)
        self.lambda_u = 0
        self.rampup_length = 16384 # get from the mixmatch codes

        for key, size in zip(["lab_loss", "unl_loss"], [500, 500]):
            self.train_dict[key] = collections.deque([], size)

    def freeze_running_stats(self, model):
        original_momentum = []
        for m_ in model.children():
            if isinstance(m_, nn.BatchNorm1d) or isinstance(m_, nn.BatchNorm2d) \
                or isinstance(m_, nn.BatchNorm3d):
                original_momentum.append(m_.momentum)
                m_.momentum = 0
        return original_momentum

    def recover_running_stats(self, model, original_momentum):
        i = 0
        for m_ in model.children():
            if isinstance(m_, nn.BatchNorm1d) or isinstance(m_, nn.BatchNorm2d) \
                or isinstance(m_, nn.BatchNorm3d):
                m_.momentum = original_momentum[i]
                i += 1
        assert(i == len(original_momentum))

    # @torch.no_grad()
    # def psuedo_label(self, unlabeled_xs):
    #     "unlabeled_xs is list of unlabeled data"
    #     K = len(unlabeled_xs)
    #     B = unlabeled_xs[0].shape[0]

    #     unlabeled_xs = torch.cat(unlabeled_xs, dim=0)

    #     p_labels = F.softmax(self.forward(unlabeled_xs, self.classifier), dim=-1)
    #     p_labels = p_labels.view(K, B, -1)
        
    #     p_label = p_labels.mean(0)

    #     return sharpening(p_label, self.hparams.T).detach()

    @torch.no_grad()
    def psuedo_label(self, unlabeled_xs):
        "unlabeled_xs is list of unlabeled data"
        p_labels = [F.softmax(self.forward(x, self.classifier), dim=-1) for x in unlabeled_xs]
        
        p_label = torch.stack(p_labels).mean(0)
        return sharpening(p_label, self.hparams.T).detach()

    def on_batch_end(self):
        # self.wdm.decay()
        self.ema.step()

        # linearly ramp up lambda_u
        if self.lambda_u == self.hparams.lambda_u:
            return

        step = (self.hparams.lambda_u - 0) / self.rampup_length
        self.lambda_u = min(self.lambda_u + step, self.hparams.lambda_u)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        labeled_x, labeled_y = batch.pop(0)
        
        labeled_y = smooth_label(labeled_y, 
                                 self.hparams.n_classes, 
                                 self.hparams.label_smoothing)

        unlabeled_xs = [b[0] for b in batch]

        B = batch_size = labeled_x.size(0)

        # not to update the running mean and variance in BN
        original_momentum = self.freeze_running_stats(self.classifier)

        p_unlabeled_y = self.psuedo_label(unlabeled_xs)

        K = len(unlabeled_xs)
        # # size [(K + 1) * B, :]
        # all_inputs = torch.cat([labeled_x] + unlabeled_xs, dim=0)
        # all_targets = torch.cat(
        #     [labeled_y, p_unlabeled_y.repeat(K, 1)],
        #     dim=0
        # )

        # mixed_input, mixed_target = half_mixup_data(all_inputs, all_targets, self.hparams.alpha)

        # logits = self.forward(mixed_input, self.classifier)

        # l_l = soft_cross_entropy(logits[:batch_size], mixed_target[:batch_size])
        # l_u = l2_distribution_loss(logits[batch_size:], mixed_target[batch_size:])

        all_inputs = torch.cat([labeled_x] + unlabeled_xs, dim=0)
        all_targets = torch.cat(
            [labeled_y, p_unlabeled_y.repeat(K, 1)],
            dim=0
        )

        mixed_input, mixed_target = half_mixup_data(all_inputs, all_targets, self.hparams.alpha)

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits_other = [self.forward(x, self.classifier) for x in mixed_input[1:]]

        self.recover_running_stats(self.classifier, original_momentum)

        # only update the BN stats for the first batch
        logits_first = self.forward(mixed_input[0], self.classifier)

        logits = [logits_first] + logits_other

        # put interleaved samples back
        logits = interleave(logits, batch_size)

        logits = torch.cat(logits, dim=0)

        l_l = soft_cross_entropy(logits[:batch_size], mixed_target[:batch_size])
        l_u = l2_distribution_loss(logits[batch_size:], mixed_target[batch_size:])

        loss = l_l + self.lambda_u * l_u

        y = torch.argmax(mixed_target, dim=-1)
        acc = self.accuracy(logits[:batch_size], y[:batch_size])


        self.train_dict["loss"].append(loss.item())
        self.train_dict["acc"].append(acc.item())
        self.train_dict["lab_loss"].append(l_l.item())
        self.train_dict["unl_loss"].append(l_u.item())

        tensorboard_logs = {"train/loss": np.mean(self.train_dict["loss"]),
                            "train/acc": np.mean(self.train_dict["acc"]),
                            "train/lab_loss": np.mean(self.train_dict["lab_loss"]),
                            "train/unl_loss": np.mean(self.train_dict["unl_loss"]),
                            "lambda_u": self.lambda_u}
        
        progress_bar = {"acc": np.mean(self.train_dict["acc"]), 
                        "lab_loss": np.mean(self.train_dict["lab_loss"]),
                        "unl_loss": np.mean(self.train_dict["unl_loss"]),
                        "lambda_u": self.lambda_u}

        return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar}

    def validation_epoch_end(self, outputs):
        # Monitor both validation set and test set
        # record the test accuracies of last 20 checkpoints

        avg_loss_list, avg_acc_list = [], []

        for output in outputs:
            avg_loss = torch.stack([x["val_loss"] * x["val_num"] for x in output]).sum() / \
                np.sum([x["val_num"] for x in output])
            avg_acc = torch.stack([x["val_acc"] * x["val_num"] for x in output]).sum() / \
                np.sum([x["val_num"] for x in output])
            
            avg_loss_list.append(avg_loss)
            avg_acc_list.append(avg_acc)

        # record best results of validation set
        self.train_dict["val_acc"][0] = max(self.train_dict["val_acc"][0], avg_acc_list[0].item())
        self.train_dict["test_acc"].append(avg_acc_list[1].item())

        tensorboard_logs = {"valid/loss": avg_loss_list[0], 
                            "valid/acc": avg_acc_list[0], 
                            "valid/best_acc": self.train_dict["val_acc"][0],
                            "test/median_acc": np.median(self.train_dict["test_acc"])}

        return {"val_loss": avg_loss_list[0], "val_acc": avg_acc_list[0], "log": tensorboard_logs}


if __name__ == "__main__":
    import torch.nn as nn
    class config:
        def __init__(self):
            self.T = 0.5
            self.n_classes = 4
            self.label_smoothing = 0
            self.ema = 0.999
            self.learning_rate = 0.1

    c = config()
    a = torch.randn(5, 3)
    y = torch.LongTensor([0, 1, 2])
    y = smooth_label(y, c.n_classes, 0)
    a_list = [torch.randn(5, 3) for _ in range(6)]
    m = nn.Linear(3, c.n_classes)

    mixmatch = NewMixmatchModule(c, m, None)

    trainer = pl.trainer.Trainer()

    trainer.fit(mixmatch)

    mixmatch.psuedo_label(a_list)
    # c.T = 1
    # print(mixmatch.psuedo_label(a_list))
    # print(mixmatch.loss(m, a, y, a_list))
