import abc
import math
import torch
import random
import itertools
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import pytorch_lightning as pl


def get_split_indices(labels, num_labeled, num_val, _n_classes):
    """
    Split the train data into the following three set:
    (1) labeled data
    (2) unlabeled data
    (3) val data

    Data distribution of the three sets are same as which of the
    original training data.

    Inputs:
        labels: (np.int) array of labels
        num_labeled: (int)
        num_val: (int)
        _n_classes: (int)


    Return:
        the three indices for the three sets
    """
    # val_per_class = num_val // _n_classes
    val_indices = []
    train_indices = []

    num_total = len(labels)
    num_per_class = []
    for c in range(_n_classes):
        num_per_class.append((labels == c).sum().astype(int))

    # obtain val indices, data evenly drawn from each class
    for c, num_class in zip(range(_n_classes), num_per_class):
        val_this_class = max(int(num_val * (num_class / num_total)), 1)
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        val_indices.append(class_indices[:val_this_class])
        train_indices.append(class_indices[val_this_class:])

    # split data into labeled and unlabeled
    labeled_indices = []
    unlabeled_indices = []

    # num_labeled_per_class = num_labeled // _n_classes

    for c, num_class in zip(range(_n_classes), num_per_class):
        num_labeled_this_class = max(int(num_labeled * (num_class / num_total)), 1)
        labeled_indices.append(train_indices[c][:num_labeled_this_class])
        unlabeled_indices.append(train_indices[c][num_labeled_this_class:])

    labeled_indices = np.hstack(labeled_indices)
    unlabeled_indices = np.hstack(unlabeled_indices)
    val_indices = np.hstack(val_indices)

    return labeled_indices, unlabeled_indices, val_indices


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

        nums = [0 for _ in range(10)]
        for i in range(len(self.indices)):
            nums[self.dataset[self.indices[i]][1]] += 1

        print(nums)
        print(np.sum(nums))

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.indices)


class MultiDataset(Dataset):
    """
    MultiDataset is used for training multiple datasets together. The lengths of the datasets
    should be the same.
    """

    def __init__(self, datasets):
        super(MultiDataset, self).__init__()
        assert len(datasets) > 1, "You should use at least two datasets"

        for d in datasets[1:]:
            assert len(d) == len(
                datasets[0]
            ), "The lengths of the datasets should be the same."

        self.datasets = datasets
        self.max_length = max([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.datasets])

    def __len__(self):
        return self.max_length


class MagicClass(object):
    """
    Codes are borrowed from https://github.com/PyTorchLightning/pytorch-lightning/pull/1959
    """

    def __init__(self, data) -> None:
        self.d = data
        self.l = max([len(d) for d in self.d])

    def __len__(self) -> int:
        return self.l

    def __iter__(self):
        if isinstance(self.d, list):
            gen = [None for v in self.d]

            # for k,v in self.d.items():
            #     # gen[k] = itertools.cycle(v)
            #     gen[k] = iter(v)

            for i in range(self.l):
                rv = [None for v in self.d]
                for k, v in enumerate(self.d):
                    # If reaching the end of the iterator, recreate one
                    # because shuffle=True in dataloader, the iter will return a different order
                    if i % len(v) == 0:
                        gen[k] = iter(v)
                    rv[k] = next(gen[k])

                yield rv

        else:
            gen = itertools.cycle(self.d)
            for i in range(self.l):
                batch = next(gen)
                yield batch


class CustomSemiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.map_indices = [[] for _ in self.datasets]
        self.min_length = min(len(d) for d in self.datasets)
        self.max_length = max(len(d) for d in self.datasets)

    def __getitem__(self, i):
        # return tuple(d[i] for d in self.datasets)

        # self.map_indices will reload when calling self.__len__()
        return tuple(d[m[i]] for d, m in zip(self.datasets, self.map_indices))

    def construct_map_index(self):
        """
        Construct the mapping indices for every data. Because the __len__ is larger than the size of some datset,
        the map_index is use to map the parameter "index" in __getitem__ to a valid index of each dataset.
        Because of the dataset has different length, we should maintain different indices for them.
        """

        def update_indices(original_indices, data_length, max_data_length):
            # update the sampling indices for this dataset

            # return: a list, which maps the range(max_data_length) to the val index in the dataset

            original_indices = original_indices[max_data_length:]  # remove used indices
            fill_num = max_data_length - len(original_indices)
            batch = math.ceil(fill_num / data_length)

            additional_indices = list(range(data_length)) * batch
            random.shuffle(additional_indices)

            original_indices += additional_indices

            assert (
                len(original_indices) >= max_data_length
            ), "the length of matcing indices is too small"

            return original_indices

        # use same mapping index for all unlabeled dataset for data consistency
        # the i-th dataset is the labeled data
        self.map_indices = [
            update_indices(m, len(d), self.max_length)
            for m, d in zip(self.map_indices, self.datasets)
        ]

        # use same mapping index for all unlabeled dataset for data consistency
        # the i-th dataset is the labeled data
        for i in range(1, len(self.map_indices)):
            self.map_indices[i] = self.map_indices[1]

    def __len__(self):
        # will be called every epoch
        return self.max_length


class DataModuleBase(pl.LightningDataModule):
    labeled_indices: ...
    unlabeled_indices: ...
    val_indices: ...

    def __init__(
        self, data_root, num_workers, batch_size, num_labeled, num_val, n_classes
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.num_val = num_val
        self._n_classes = n_classes

        self.train_transform = None  # TODO, need implement this in your custom datasets
        self.test_transform = None  # TODO, need implement this in your custom datasets

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.num_workers = num_workers

    def train_dataloader(self):
        # get and process the data first

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # return both val and test loader

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return [val_loader, test_loader]

    def test_dataloader(self):

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def n_classes(self):
        # self._n_class should be defined in _prepare_train_dataset()
        return self._n_classes

    @property
    def num_labeled_data(self):
        assert self.train_set is not None, (
            "Load train data before calling %s" % self.num_labeled_data.__name__
        )
        return len(self.labeled_indices)

    @property
    def num_unlabeled_data(self):
        assert self.train_set is not None, (
            "Load train data before calling %s" % self.num_unlabeled_data.__name__
        )
        return len(self.unlabeled_indices)

    @property
    def num_val_data(self):
        assert self.train_set is not None, (
            "Load train data before calling %s" % self.num_val_data.__name__
        )
        return len(self.val_indices)

    @property
    def num_test_data(self):
        assert self.test_set is not None, (
            "Load test data before calling %s" % self.num_test_data.__name__
        )
        return len(self.test_set)


class SemiDataModule(DataModuleBase):
    """
    Data module for semi-supervised tasks. self.prepare_data() is not implemented. For custom dataset,
    inherit this class and implement self.prepare_data().
    """

    def __init__(
        self,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        num_augments,
        n_classes,
    ):
        super(SemiDataModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )
        self.num_augments = num_augments

    def setup(self):
        # prepare train and val dataset, and split the train dataset
        # into labeled and unlabeled groups.
        assert (
            self.train_set is not None
        ), "Should create self.train_set in self.setup()"

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        # np.random.shuffle(ys)
        # get the number of classes
        # self._n_classes = len(np.unique(ys))

        (
            self.labeled_indices,
            self.unlabeled_indices,
            self.val_indices,
        ) = get_split_indices(ys, self.num_labeled, self.num_val, self._n_classes)

        self.val_set = Subset(self.train_set, self.val_indices, self.test_transform)

        # unlabeled_list = [
        #     Subset(self.train_set, self.unlabeled_indices, self.train_transform) \
        #         for _ in range(self.num_augments)
        # ]

        # self.unlabeled_set = MultiDataset(unlabeled_list)
        # self.labeled_set = Subset(self.train_set, self.labeled_indices, self.train_transform)

        train_list = [
            Subset(self.train_set, self.unlabeled_indices, self.train_transform)
            for _ in range(self.num_augments)
        ]

        train_list.insert(
            0, Subset(self.train_set, self.labeled_indices, self.train_transform)
        )

        self.train_set = CustomSemiDataset(train_list)

    # def train_dataloader(self):
    #     # get and process the data first
    #     if self.labeled_set is None:
    #         self._prepare_train_dataset()

    #     labeled_loader = DataLoader(self.labeled_set,
    #                                 batch_size=self.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=self.num_workers,
    #                                 pin_memory=True,
    #                                 drop_last=True)

    #     unlabeled_loader = DataLoader(self.unlabeled_set,
    #                                 batch_size=self.batch_size,
    #                                 shuffle=True,
    #                                 num_workers=self.num_workers,
    #                                 pin_memory=True,
    #                                 drop_last=True)

    #     return MagicClass([labeled_loader, unlabeled_loader])

    def train_dataloader(self):
        # get and process the data first

        self.train_set.construct_map_index()

        print("\ncalled\n")

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class SupervisedDataModule(DataModuleBase):
    """
    Data module for supervised tasks. self.prepare_data() is not implemented. For custom dataset,
    inherit this class and implement self.prepare_data().
    """

    def __init__(
        self, data_root, num_workers, batch_size, num_labeled, num_val, n_classes
    ):
        super(SupervisedDataModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

    def setup(self):
        # prepare train and val dataset
        assert (
            self.train_set is not None
        ), "Should create self.train_set in self.setup()"

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        # get the number of classes
        # self._n_classes = len(np.unique(ys))

        (
            self.labeled_indices,
            self.unlabeled_indices,
            self.val_indices,
        ) = get_split_indices(ys, self._n_classes, self.num_val, self._n_classes)

        self.labeled_indices = np.hstack((self.labeled_indices, self.unlabeled_indices))
        self.unlabeled_indices = []  # dummy. only for printing length

        self.val_set = Subset(self.train_set, self.val_indices, self.test_transform)
        self.train_set = Subset(
            self.train_set, self.labeled_indices, self.train_transform
        )
