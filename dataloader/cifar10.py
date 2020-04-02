import numpy as np
from .basedata import SemiDataLoader, SupervisedDataLoader

import torchvision as tv
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10

class SemiCIFAR10Loader(SemiDataLoader):
    def __init__(self, args, data_root, num_workers, batch_size, num_labeled, valid_percent):
        super(SemiCIFAR10Loader, self).__init__(data_root, num_workers, batch_size, num_labeled, valid_percent)
        
        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
  
        self.test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])

        self._prepare_train_dataset()

    def load_dataset(self, is_train=True):
        # the transformation for train and validation dataset will be 
        # done in _prepare_train_dataset()
        if is_train:
            return CIFAR10(self.data_root, train=True, download=True, 
                           transform=None)

        return CIFAR10(self.data_root, train=False, download=True, 
            transform=self.test_transform)

class SupervisedCIFAR10Loader(SupervisedDataLoader):
    def __init__(self, args, data_root, num_workers, batch_size, num_labeled, valid_percent):
        super(SupervisedCIFAR10Loader, self).__init__(data_root, num_workers, batch_size, num_labeled, valid_percent)
        

        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        
        self.test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])

        self._prepare_train_dataset()

    def load_dataset(self, is_train=True):
        # the transformation for train and validation dataset will be 
        # done in _prepare_train_dataset()
        if is_train:
            return CIFAR10(self.data_root, train=True, download=True, 
                           transform=None)

        return CIFAR10(self.data_root, train=False, download=True, 
            transform=self.test_transform)