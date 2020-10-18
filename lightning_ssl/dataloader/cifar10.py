import numpy as np
from .base_data import SemiDataLoader, SupervisedDataLoader

import torchvision as tv
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)

class SemiCIFAR10Loader(SemiDataLoader):
    def __init__(self, args, data_root, num_workers, batch_size, num_labeled, num_val, num_augments):
        super(SemiCIFAR10Loader, self).__init__(data_root, num_workers, batch_size, num_labeled, num_val, num_augments)
        
        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
  
        self.test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
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
    def __init__(self, args, data_root, num_workers, batch_size, num_labeled, num_val, num_augments):
        super(SupervisedCIFAR10Loader, self).__init__(data_root, num_workers, batch_size, num_labeled, num_val)
        
        self.train_transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        
        self.test_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
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


if __name__ == "__main__":
    np.random.seed(1)
    t = SupervisedCIFAR10Loader(None, 
        "/work/ntubiggg1/dataset", 
        16, 64, 250, 500, 2)