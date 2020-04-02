import abc
import torch
import random
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset

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

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.indices)

class CustomDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

        self.map_indexes = [[] for _ in self.datasets]
        self.min_length = min(len(d) for d in self.datasets)
        self.max_length = max(len(d) for d in self.datasets)

    def __getitem__(self, i):
        # return tuple(d[i] for d in self.datasets)

        # self.map_indexes will reload when calling self.__len__()
        return tuple(d[m[i]] for d, m in zip(self.datasets, self.map_indexes))

    def construct_map_index(self):
        def update_indices(original_indexes, target_len, max_len):
            # map max_len to target_len (large to small)

            # return: a list, which maps the range(max_len) to the valid index in the dataset
            
            original_indexes = original_indexes[max_len:] # remove used indices
            fill_num = max_len - len(original_indexes)
            batch = fill_num // target_len

            if fill_num % target_len != 0:
                # to let the fill_num + len(original_indexes) greater than max_len
                batch += 1

            # batch = max_len // target_len
            # residual = max_len % target_len

            additional_indexes = list(range(target_len)) * batch
            random.shuffle(additional_indexes)

            original_indexes += additional_indexes

            # indexes = base_indexes * batch + random.sample(base_indexes, residual)

            assert len(original_indexes) >= max_len, "the length of matcing indexes is too small"

            return original_indexes

        self.map_indexes = [update_indices(m, len(d), self.max_length) 
            for m, d in zip(self.map_indexes, self.datasets)]

    def __len__(self):
        # will be called every epoch
        self.construct_map_index()
        return self.max_length

class AbstractDataLoader(abc.ABC):
    def __init__(self, data_root, num_workers, batch_size, num_labeled, valid_percent):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_labeled = num_labeled
        self.valid_percent = valid_percent

        self.train_transform = None # TODO, need implement this in your custom datasets
        self.test_transform = None # TODO, need implement this in your custom datasets

        self.train_set = None
        self.valid_set = None
        self.test_set = None

        self.num_workers = num_workers

    @abc.abstractmethod
    def _prepare_train_dataset(self):
        # process train dataset
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
    def num_valid_data(self):
        assert self.train_set is not None, \
            "Load train data before calling %s" % self.num_valid_data.__name__
        return len(self.valid_indices)

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
                          num_workers=self.num_workers)
    
    def get_valid_loader(self):
        # get and process the data first
        if self.valid_set is None:
            self._prepare_train_dataset()
        
        return DataLoader(self.valid_set,
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers=self.num_workers)

    def get_test_loader(self):
        # get and process the data first
        if self.test_set is None:
            self._prepare_test_dataset()

        return DataLoader(self.test_set,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

class SemiDataLoader(AbstractDataLoader):
    def __init__(self, data_root, num_workers, batch_size, num_labeled, valid_percent, num_augments):
        super(SemiDataLoader, self).__init__(data_root, num_workers, batch_size, num_labeled, valid_percent)
        self.num_augments = num_augments

    def _prepare_train_dataset(self):
        # prepare train and valid dataset, and split the train dataset 
        # into labeled and unlabeled groups.
        self.train_set = self.load_dataset(is_train=True)

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        np.random.shuffle(ys)
        # get the number of classes
        self._n_classes = len(np.unique(ys))
        
        valid_size = int(self.valid_percent * len(self.train_set))
        valid_per_class = valid_size // self._n_classes
        valid_indices = []
        train_indices = []

        # obtain valid indices, data evenly drawn from each class
        for c in range(self._n_classes):
            class_indices = np.where(ys == c)[0]
            valid_indices.append(class_indices[: valid_per_class])
            train_indices.append(class_indices[valid_per_class:])

        # split data into labeled and unlabeled
        labeled_indices = []
        unlabeled_indices = []

        num_labeled_per_class = self.num_labeled // self._n_classes

        for c in range(self._n_classes):
            labeled_indices.append(train_indices[c][:num_labeled_per_class])
            unlabeled_indices.append(train_indices[c][num_labeled_per_class:])

        self.labeled_indices = np.hstack(labeled_indices)
        self.unlabeled_indices = np.hstack(unlabeled_indices)
        self.valid_indices = np.hstack(valid_indices)

        self.valid_set = Subset(self.train_set, self.valid_indices, self.test_transform)

        train_list = [
            Subset(self.train_set, self.unlabeled_indices, self.train_transform) \
                for _ in range(self.num_augments)
        ]

        train_list.insert(0, Subset(self.train_set, self.labeled_indices, self.train_transform))

        self.train_set = CustomDataset(train_list)

class SupervisedDataLoader(AbstractDataLoader):
    def __init__(self, data_root, num_workers, batch_size, num_labeled, valid_percent):
        super(SupervisedDataLoader, self).__init__(data_root, num_workers, batch_size, num_labeled, valid_percent)

    def _prepare_train_dataset(self):
        # prepare train and valid dataset, and split the train dataset 
        # into labeled and unlabeled groups.
        self.train_set = self.load_dataset(is_train=True)

        indices = np.arange(len(self.train_set))
        ys = np.array([self.train_set[i][1] for i in indices], dtype=np.int64)
        np.random.shuffle(ys)
        # get the number of classes
        self._n_classes = len(np.unique(ys))
        
        valid_size = int(self.valid_percent * len(self.train_set))
        valid_per_class = valid_size // self._n_classes
        valid_indices = []
        train_indices = []

        # obtain valid indices, data evenly drawn from each class
        for c in range(self._n_classes):
            class_indices = np.where(ys == c)[0]
            valid_indices.append(class_indices[: valid_per_class])
            train_indices.append(class_indices[valid_per_class:])

        self.labeled_indices = np.hstack(train_indices)
        self.valid_indices = np.hstack(valid_indices)
        self.unlabeled_indices = [] # dummy. only for printing length

        self.valid_set = Subset(self.train_set, self.valid_indices, self.test_transform)
        self.train_set = Subset(self.train_set, self.labeled_indices, self.train_transform)

if __name__ == "__main__":
        
    from torch.utils.data import TensorDataset
            
    dataset_1 = TensorDataset(torch.arange(2))
    dataset_2 = TensorDataset(torch.arange(3, 8))

    dataset = CustomDataset([dataset_1, dataset_2])

    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    print("start epoch")
    for i in range(3):
        for batch in dataloader:
            print(batch)
            print("="*5)