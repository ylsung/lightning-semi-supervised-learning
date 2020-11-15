import random
import numpy as np
from lightning_ssl.dataloader.base_data import (
    SemiDataLoader,
    SupervisedDataLoader,
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)


def sample_dataset():
    return [
        (np.random.uniform(0, 1, (3, 2, 2)).astype(np.float32), random.randint(0, 9))
        for i in range(50)
    ]


class SemiSampleModule(SemiDataModule):
    def __init__(
        self,
        args,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        num_augments,
    ):
        n_classes = 10
        super(SemiSampleModule, self).__init__(
            data_root,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            num_augments,
            n_classes,
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = sample_dataset()
        self.test_set = sample_dataset()


class SupervisedSampleModule(SupervisedDataModule):
    def __init__(
        self,
        args,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        num_augments,
    ):
        n_classes = 10
        super(SupervisedSampleModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = sample_dataset()
        self.test_set = sample_dataset()
