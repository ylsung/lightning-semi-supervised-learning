import torch
import pytest

from copy import deepcopy

from lightning_ssl.models import WideResNet


@pytest.mark.parametrize(
    "shape, num_classes", [((5, 3, 32, 32), 8), ((10, 3, 32, 32), 15)]
)
def test_wideresnet(shape, num_classes):
    inputs = torch.randn(shape)
    model = WideResNet(28, num_classes, dropRate=0.2)
    batch_size = inputs.shape[0]
    assert model(inputs).shape == torch.Size([batch_size, num_classes])
