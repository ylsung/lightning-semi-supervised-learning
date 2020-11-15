import unittest
import torch
import torch.nn
import pytest

from copy import deepcopy

from lightning_ssl.models import WideResNet


@pytest.fixture(scope="module")
def model_inputs_pair():
    return WideResNet(28, 10), torch.randn((5, 3, 32, 32))


def test_pseudo_label(model_inputs_pair):
    model = WideResNet(28, 10)
    model.eval()
    inputs_list = [torch.randn(5, 3, 32, 32) for _ in range(3)]

    assert model.psuedo_label(inputs_list, 0.5).shape == torch.Size([5, 10])

    assert torch.all(
        torch.isclose(
            model.psuedo_label(inputs_list, 0.5),
            model.psuedo_label(inputs_list, 0.5, True),
        )
    )


def test_freeze_running_stats(model_inputs_pair):
    model, inputs = model_inputs_pair
    model.freeze_running_stats()
    before_stats = deepcopy(model.extract_running_stats())
    # run the network
    model(inputs)
    # the running_stats should not change
    after_stats = deepcopy(model.extract_running_stats())

    # check the number of batch norm layers
    assert len(before_stats) == (28 - 4) // 6 * 2 * 3 + 1

    for f, s in zip(before_stats, after_stats):
        assert torch.all(torch.eq(f[0], s[0]))
        assert torch.all(torch.eq(f[1], s[1]))


def test_recover_running_stats(model_inputs_pair):
    model, inputs = model_inputs_pair
    # recover running stats
    model.recover_running_stats()
    before_stats = deepcopy(model.extract_running_stats())
    # run the network
    model(inputs)

    # the running_stats should not change
    after_stats = deepcopy(model.extract_running_stats())

    for f, s in zip(before_stats, after_stats):
        assert torch.all(~torch.eq(f[0], s[0]))
        assert torch.all(~torch.eq(f[1], s[1]))


def test_recover_running_stats__without_freeze_first():
    model = WideResNet(28, 10)
    before_stats = model.extract_norm_n_momentum()
    model.recover_running_stats()
    after_stats = model.extract_norm_n_momentum()

    # test batchnorm statistics
    for f, s in zip(before_stats[0], after_stats[0]):
        assert torch.all(torch.eq(f.running_mean, s.running_mean))
        assert torch.all(torch.eq(f.running_var, s.running_var))
    # test momentum
    for f, s in zip(before_stats[1], after_stats[1]):
        assert f == s
