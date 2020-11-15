import torch
import pytest
import numpy as np
from lightning_ssl.dataloader.base_data import (
    get_split_indices,
    CustomSemiDataset,
    MultiDataset,
    MagicClass,
)
from torch.utils.data import TensorDataset, DataLoader


@pytest.fixture(
    params=[
        {"max_label": 10, "num_data": 1000},
        {"max_label": 10, "num_data": 1000},
        {"max_label": 10, "num_data": 1000},
    ]
)
def label_data(request):
    param = request.param

    while True:
        label_proportion = np.random.uniform(
            0, 1, size=(param["max_label"],)
        )  # the proportion of each label

        label_proportion = (
            label_proportion / label_proportion.sum()
        )  # normalize to summation of proportion is 1

        # after this normalization
        label_proportion = np.round_(
            label_proportion, decimals=int(np.log10(param["num_data"]) - 1)
        )

        residual = 1.0 - label_proportion.sum()
        # add the residual to the last element
        label_proportion[-1] += residual

        if label_proportion.min() > 0:  # valid proportion

            label_proportion = (label_proportion * param["num_data"]).astype(int)

            if label_proportion.sum() == param["num_data"]:  # valid proportion
                label_data = []

                for i, p in enumerate(label_proportion):
                    label_data.append(i * np.ones((p,)))

                label_data = np.hstack(label_data)

                np.random.shuffle(label_data)

                return label_data


def test_get_split_indices(label_data):
    # label_data is specially design
    # the proportion of the labeled in label_data is specially designed to that
    # (num_labeled * proportion), (num_val * proportion) and (num_unlabeled * proportion)
    # are all integer

    num_labeled = int(0.1 * len(label_data))
    num_val = int(0.1 * len(label_data))
    n_classes = int(np.max(label_data).item()) + 1
    num_unlabeled = len(label_data) - num_labeled - num_val
    labeled_indices, unlabeled_indices, val_indices = get_split_indices(
        label_data, num_labeled, num_val, n_classes
    )

    def convert_idx_to_num(indices):
        num_list = []
        for c in range(n_classes):
            num_list.append((label_data[labeled_indices] == c).sum())
        return num_list

    assert len(val_indices) == num_val
    assert len(labeled_indices) == num_labeled
    assert len(unlabeled_indices) == num_unlabeled

    # check the proportion of data
    assert convert_idx_to_num(labeled_indices) == convert_idx_to_num(
        np.arange(len(label_data))
    )
    assert convert_idx_to_num(val_indices) == convert_idx_to_num(
        np.arange(len(label_data))
    )
    assert convert_idx_to_num(unlabeled_indices) == convert_idx_to_num(
        np.arange(len(label_data))
    )


def test_custum_dataset():
    dataset_1 = TensorDataset(torch.arange(10))
    dataset_2 = TensorDataset(torch.arange(30, 75))
    dataset_3 = TensorDataset(torch.arange(30, 75))
    dataset_4 = TensorDataset(torch.arange(30, 75))

    dloader_1 = DataLoader(dataset_1, batch_size=3, shuffle=True, num_workers=0)
    dloader_2 = DataLoader(
        MultiDataset([dataset_2, dataset_3, dataset_4]),
        batch_size=3,
        shuffle=True,
        num_workers=0,
    )

    for _ in range(10):
        for batch in MagicClass([dloader_1, dloader_2]):
            batch = batch[1]
            for b in batch[1:]:
                assert torch.all(
                    batch[0][0] == b[0]
                ), "The data in multidataset should be the same"
