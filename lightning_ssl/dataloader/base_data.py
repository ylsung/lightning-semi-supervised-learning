import abc
import torch
import random
import itertools
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

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
            assert len(d) == len(datasets[0]), "The lengths of the datasets should be the same."

        self.datasets = datasets
        self.max_length = max([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.datasets])

    def __len__(self):
        return self.max_length

def get_len(d):
    if isinstance(d, dict):
        v = max(d.items(), key=lambda x: len(x[1]))
        return len(v[1])
    else:
        return len(d)

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

        self.map_indices = self.construct_map_index()
        # use same mapping index for all unlabeled dataset for data consistency
        # the i-th dataset is the labeled data
        for i in range(1, len(self.map_indices)):
            self.map_indices[i] = self.map_indices[1]

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
        print("called")
        def update_indices(original_indices, data_length, max_data_length):
            # update the sampling indices for this dataset

            # return: a list, which maps the range(max_data_length) to the val index in the dataset
            
            original_indices = original_indices[max_data_length:] # remove used indices
            fill_num = max_data_length - len(original_indices)
            batch = fill_num // data_length

            if fill_num % data_length != 0:
                # to let the fill_num + len(original_indices) greater than max_data_length
                batch += 1

            # batch = max_len // target_len
            # residual = max_len % target_len

            additional_indices = list(range(data_length)) * batch
            random.shuffle(additional_indices)

            original_indices += additional_indices

            # indices = base_indices * batch + random.sample(base_indices, residual)

            assert len(original_indices) >= max_data_length, "the length of matcing indices is too small"

            return original_indices

        return [update_indices(m, len(d), self.max_length) for m, d in zip(self.map_indices, self.datasets)]


    def __len__(self):
        # will be called every epoch
        return self.max_length

class AbstractDataLoader(abc.ABC):
    _n_classes: int
    labeled_indices: ...
    unlabeled_indices: ...
    val_indices: ...

    def __init__(self, data_root, num_workers, batch_size, num_labeled, num_val):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.num_val = num_val

        self.train_transform = None # TODO, need implement this in your custom datasets
        self.test_transform = None # TODO, need implement this in your custom datasets

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.num_workers = num_workers

    @abc.abstractmethod
    def _prepare_train_dataset(self):
        # process train dataset
        return

    @abc.abstractmethod
    def load_dataset(self, *args, **kwargs):
        # load the dataset
        return
    
    @property
    def n_classes(self):
        # self._n_class should be defined in _prepare_train_dataset()
        return self._n_classes

    @property
    def num_labeled_data(self):
        assert self.train_set is not None, \
            "Load train data before calling %s" % self.num_labeled_data.__name__
        return len(self.labeled_indices)
    
    @property
    def num_unlabeled_data(self):
        assert self.train_set is not None, \
            "Load train data before calling %s" % self.num_unlabeled_data.__name__
        return len(self.unlabeled_indices)

    @property
    def num_val_data(self):
        assert self.train_set is not None, \
            "Load train data before calling %s" % self.num_val_data.__name__
        return len(self.val_indices)

    @property
    def num_test_data(self):
        assert self.test_set is not None, \
            "Load test data before calling %s" % self.num_test_data.__name__
        return len(self.test_set)

    def _prepare_test_dataset(self):
        self.test_set = self.load_dataset(is_train=False)
        # Do something if needing the processing. 
        # ...
        # ...

    def get_train_loader(self):
        # get and process the data first
        if self.train_set is None:
            self._prepare_train_dataset()

        return DataLoader(self.train_set,
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True)
    
    def get_val_loader(self):
        # get and process the data first
        if self.val_set is None:
            self._prepare_train_dataset()
        
        return DataLoader(self.val_set,
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def get_test_loader(self):
        # get and process the data first
        if self.test_set is None:
            self._prepare_test_dataset()

        return DataLoader(self.test_set,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          pin_memory=True)

class SemiDataLoader(AbstractDataLoader):
    def __init__(self, data_root, num_workers, batch_size, num_labeled, num_val, num_augments):
        super(SemiDataLoader, self).__init__(data_root, num_workers, batch_size, num_labeled, num_val)
        self.num_augments = num_augments

    def _prepare_train_dataset(self):
        # prepare train and val dataset, and split the train dataset 
        # into labeled and unlabeled groups.
        self.train_set = self.load_dataset(is_train=True)

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        # np.random.shuffle(ys)
        # get the number of classes
        self._n_classes = len(np.unique(ys))

        self.labeled_indices, self.unlabeled_indices, self.val_indices = \
            get_split_indices(ys, self.num_labeled, self.num_val, self._n_classes)

        self.val_set = Subset(self.train_set, self.val_indices, self.test_transform)

        unlabeled_list = [
            Subset(self.train_set, self.unlabeled_indices, self.train_transform) \
                for _ in range(self.num_augments)
        ]

        # train_list.insert(0, Subset(self.train_set, self.labeled_indices, self.train_transform))

        # self.train_set = CustomSemiDataset(train_list)
        self.unlabeled_set = MultiDataset(unlabeled_list)
        self.labeled_set = Subset(self.train_set, self.labeled_indices, self.train_transform)

    def get_train_loader(self):
        # get and process the data first
        if self.labeled_set is None:
            self._prepare_train_dataset()
        
        labeled_loader = DataLoader(self.labeled_set,
                                    batch_size=self.batch_size, 
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    drop_last=True)

        unlabeled_loader = DataLoader(self.unlabeled_set,
                                    batch_size=self.batch_size, 
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True,
                                    drop_last=True)
        
        return MagicClass([labeled_loader, unlabeled_loader])

class SupervisedDataLoader(AbstractDataLoader):
    def __init__(self, data_root, num_workers, batch_size, num_labeled, num_val):
        super(SupervisedDataLoader, self).__init__(data_root, num_workers, batch_size, num_labeled, num_val)

    def _prepare_train_dataset(self):
        # prepare train and val dataset
        self.train_set = self.load_dataset(is_train=True)

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        # get the number of classes
        self._n_classes = len(np.unique(ys))
        
        self.labeled_indices, self.unlabeled_indices, self.val_indices = \
            get_split_indices(ys, self._n_classes, self.num_val, self._n_classes)

        self.labeled_indices = np.hstack((self.labeled_indices, self.unlabeled_indices))
        self.unlabeled_indices = [] # dummy. only for printing length

        self.val_set = Subset(self.train_set, self.val_indices, self.test_transform)
        self.train_set = Subset(self.train_set, self.labeled_indices, self.train_transform)

if __name__ == "__main__":
        
    from torch.utils.data import TensorDataset
            
    dataset_1 = TensorDataset(torch.arange(2))
    dataset_2 = TensorDataset(torch.arange(3, 8))
    dataset_3 = TensorDataset(torch.arange(3, 8))
    dataset_4 = TensorDataset(torch.arange(3, 8))

    dataset = CustomSemiDataset([dataset_1, dataset_2, dataset_3, dataset_4])
    
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    print("start epoch")
    for i in range(3):
        for batch in dataloader:
            print(batch)
            print("="*5)