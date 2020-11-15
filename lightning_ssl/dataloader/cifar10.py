import numpy as np
from lightning_ssl.dataloader.base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)


class SemiCIFAR10Module(SemiDataModule):
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
        super(SemiCIFAR10Module, self).__init__(
            data_root,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            num_augments,
            n_classes,
        )

        self.train_transform = tv.transforms.Compose(
            [
                tv.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        self.test_transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = CIFAR10(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = CIFAR10(
            self.data_root, train=False, download=True, transform=self.test_transform
        )


class SupervisedCIFAR10Module(SupervisedDataModule):
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
        super(SupervisedCIFAR10Module, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose(
            [
                tv.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        self.test_transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = CIFAR10(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = CIFAR10(
            self.data_root, train=False, download=True, transform=self.test_transform
        )
