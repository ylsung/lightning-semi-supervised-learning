import unittest
import pytest
import torch
import numpy as np
import torchvision
from copy import deepcopy

from lightning_ssl.utils.torch_utils import (
    sharpening,
    soft_cross_entropy,
    l2_distribution_loss,
    mixup_data,
    smooth_label,
    to_one_hot_vector,
    mixup_data_for_testing,
    half_mixup_data_for_testing,
    customized_weight_decay,
    split_weight_decay_weights,
    interleave,
    my_interleave,
    WeightDecayModule,
)

from lightning_ssl.models import WideResNet


@pytest.fixture
def logits_targets_pair():
    """
    The fixture to test softmax related function. Return the logits and soft targets.
    Return
        logits, targets
    """
    logits = [[-1, 1, 0], [1, 0, -1]]
    targets = [[0.3, 0.6, 0.1], [0.5, 0.2, 0.3]]
    return torch.FloatTensor(logits), torch.FloatTensor(targets)


@pytest.fixture(
    params=[
        {"batch_size": 1, "num_classes": 5, "smoothing_factor": 0.3},
        {"batch_size": 3, "num_classes": 5, "smoothing_factor": 0.01},
        {"batch_size": 3, "num_classes": 5, "smoothing_factor": 0.0},
        {"batch_size": 3, "num_classes": 5, "smoothing_factor": 1},
    ]
)
def targets_smoothing_classes_tuple(request):
    param = request.param
    return (
        torch.randint(param["num_classes"], size=(param["batch_size"],)),
        param["smoothing_factor"],
        param["num_classes"],
    )


@pytest.fixture(
    params=[
        {"batch_size": 20, "num_classes": 10},
        {"batch_size": 30, "num_classes": 2},
        {"batch_size": 20, "num_classes": 100},
    ]
)
def images_targets_num_classes_tuple(request):
    param = request.param
    batch_size = param["batch_size"]
    num_classes = param["num_classes"]
    return (
        torch.randn(batch_size, 3, 32, 32),
        torch.randint(num_classes, size=(batch_size,)),
        num_classes,
    )


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1, 2, 10])
def test_sharpening(temperature):
    def entropy(probs):
        return torch.sum(-probs * torch.log(probs), -1).mean()

    for _ in range(10):
        logits = torch.randn(4, 10)
        probs = torch.softmax(logits, -1)
        sharpening_probs = sharpening(probs, temperature)

        if temperature > 1:
            assert entropy(sharpening_probs) > entropy(probs)
        elif temperature < 1:
            assert entropy(sharpening_probs) < entropy(probs)
        else:
            assert torch.isclose(entropy(sharpening_probs), entropy(probs))


def test_soft_cross_entropy(logits_targets_pair):
    logits, targets = logits_targets_pair

    probs = torch.softmax(logits, -1)

    ans = 0
    for i in range(logits.shape[0]):
        for c in range(logits.shape[1]):
            ans += -torch.log(probs[i][c]) * targets[i][c]

    ans /= logits.shape[0]
    func_out = soft_cross_entropy(logits, targets, dim=-1)

    assert ans == func_out


def test_l2_distribution_loss(logits_targets_pair):
    logits, targets = logits_targets_pair

    probs = torch.softmax(logits, -1)

    ans = 0
    for i in range(logits.shape[0]):
        for c in range(logits.shape[1]):
            ans += (probs[i][c] - targets[i][c]) ** 2

    ans /= logits.shape[0] * logits.shape[1]
    func_out = l2_distribution_loss(logits, targets, dim=-1)

    assert ans == func_out


def test_smooth_label(targets_smoothing_classes_tuple):
    targets, smoothing_factor, num_classes = targets_smoothing_classes_tuple
    ##################################################################
    # test for label smoothing
    batch_size = targets.shape[0]
    y_labels = smooth_label(targets, num_classes, smoothing_factor)

    # predicted classes
    pred = torch.argmax(y_labels, dim=-1)

    # the logits of maximum classes should be (1 - smoothing_factor)
    assert torch.all(
        y_labels[torch.arange(batch_size), targets] == 1 - smoothing_factor
    )
    # all the other logits should be the same
    assert torch.sum(y_labels != 1 - smoothing_factor) == batch_size * (num_classes - 1)
    # summation should be equal to 1
    assert torch.all(torch.isclose(torch.sum(y_labels, -1), torch.ones(pred.shape)))


def test_one_hot(targets_smoothing_classes_tuple):
    targets, _, num_classes = targets_smoothing_classes_tuple
    batch_size = targets.shape[0]
    # test for one hot transformation
    y_labels = to_one_hot_vector(targets, num_classes)

    # predicted classes
    pred = torch.argmax(y_labels, dim=-1)

    # the logits of maximum classes should be 1
    assert torch.all(y_labels[torch.arange(batch_size), pred] == 1)
    # the maximum classes should be the same as targets
    assert torch.all(torch.eq(pred, targets))
    # summation should be equal to 1
    assert torch.all(torch.isclose(torch.sum(y_labels, -1), torch.ones(pred.shape)))


def test_mixup_minus(images_targets_num_classes_tuple):
    inputs, targets, _ = images_targets_num_classes_tuple
    mixed_inputs, mixed_targets = mixup_data(inputs, targets, -1)

    assert torch.all(torch.eq(inputs, mixed_inputs))
    assert torch.all(torch.eq(targets, mixed_targets))


def test_mixup(images_targets_num_classes_tuple):
    inputs, targets, num_classes = images_targets_num_classes_tuple

    logits = smooth_label(targets, num_classes)

    _, _, _, _, gt_lambda = mixup_data_for_testing(inputs, logits, 0.5)

    # the mixed data should be between it's two ingredients

    # exclude the data which shuffle to the same place,
    # this will cause mixed data equal to the origin ingredients,
    # but sometimes two same numbers will be considered different due to the limitation float number

    # test lambda is within the range
    assert torch.all((gt_lambda >= 0) * (gt_lambda <= 1))

    # same_pos_x = torch.all(torch.isclose(inputs, p_inputs).reshape(batch_size, -1), dim=-1)

    # inputs, p_inputs, logits, p_logits = \
    #     inputs[~same_pos_x], p_inputs[~same_pos_x], logits[~same_pos_x], p_logits[~same_pos_x]

    # mixed_x, mixed_y = mixed_x[~same_pos_x], mixed_y[~same_pos_x]

    # min_x, max_x = torch.min(inputs, p_inputs), torch.max(inputs, p_inputs)
    # min_y, max_y = torch.min(logits, p_logits), torch.max(logits, p_logits)

    # pos = ~((min_x <= mixed_x) * (mixed_x <= max_x))

    # print(inputs[pos]==mixed_x[pos])
    # print(p_inputs[pos])

    # print(mixed_x[pos])

    # assert torch.all((min_x <= mixed_x) * (mixed_x <= max_x)) == True

    # def reconstruct(mixed_d, original_d, shuffle_d):
    #     numerator = mixed_d - shuffle_d
    #     denominator = original_d - shuffle_d
    #     mask = (denominator == 0).float()

    #     reverted_l = numerator / (denominator + mask * 1e-15)

    #     return reverted_l

    # l_1 = reconstruct(mixed_x, inputs, p_inputs)
    # l_1 = torch.mode(l_1.reshape(l_1.shape[0], -1), -1)[0]
    # l_2 = reconstruct(mixed_y, y_labels, p_y_labels)
    # l_2 = torch.max(l_2.reshape(l_2.shape[0], -1), -1)[0]

    # for i in range(l_1.shape[0]):
    #     # only check when the y_label and permuted y_label are different,
    #     # or the value will be 0
    #     if torch.all(y_labels[i] == p_y_labels[i]):
    #         assert l_2[i] == 0

    #     elif torch.all(inputs[i] == p_inputs[i]):
    #             assert l_1[i] == 0
    #     else:
    #         assert torch.isclose(l_1[i], l_2[i], atol=1e-5) == 0
    #         assert 0.0 <= l_1[i] <= 1 == True


def test_half_mixup(images_targets_num_classes_tuple):
    inputs, targets, num_classes = images_targets_num_classes_tuple

    logits = smooth_label(targets, num_classes)

    _, _, _, _, gt_lambda = half_mixup_data_for_testing(inputs, logits, 0.5)

    # test lambda is within the range
    assert torch.all((gt_lambda >= 0.5) * (gt_lambda <= 1))


def test_customized_weight_decay():
    m = torchvision.models.resnet34()
    m_copy = deepcopy(m)

    ignore_key = ["bn", "bias"]
    wd = 1e-2
    customized_weight_decay(m, wd, ignore_key=ignore_key)

    for (name_m, p_m), (name_copy, p_copy) in zip(
        m.named_parameters(), m_copy.named_parameters()
    ):
        assert name_m == name_copy
        # ignore, and don't decay weight
        if any(key in name_m for key in ignore_key):
            assert torch.all(torch.eq(p_m, p_copy))
        else:
            assert torch.all(torch.eq(p_m, p_copy * (1 - wd)))


def test_weight_decay_module():
    m = WideResNet(28, 10)

    wd_module = WeightDecayModule(m, 0.1, ["bn", "bias"])

    num_weights = 29  # weights of conv and fc

    assert len(wd_module.available_parameters) == num_weights

    wd_module = WeightDecayModule(m, 0.1, ["weight"])

    num_bias = 25 + 1 + 0  # bias of bn + bias of fc + bias of conv

    assert len(wd_module.available_parameters) == num_bias

    original_parameters = deepcopy(wd_module.available_parameters)
    wd_module.decay()

    for original_weight, decay_weight in zip(
        original_parameters, wd_module.available_parameters
    ):
        assert torch.all(torch.eq(original_weight * 0.9, decay_weight))


def test_split_weight_decay_weights():
    m = torchvision.models.resnet34()
    m_copy = deepcopy(m)

    ignore_key = ["bn", "bias"]
    wd = 1e-2
    customized_weight_decay(m, wd, ignore_key=ignore_key)

    num_weight_decay = 0
    num_no_weight_decay = 0
    for (name_m, _), (name_copy, _) in zip(
        m.named_parameters(), m_copy.named_parameters()
    ):
        assert name_m == name_copy
        # ignore, and don't decay weight
        if any(key in name_m for key in ignore_key):
            num_no_weight_decay += 1
        else:
            num_weight_decay += 1

    params_list = split_weight_decay_weights(m, wd, ignore_key=ignore_key)

    # first item is no weight decay

    assert len(params_list[0]["params"]) == num_no_weight_decay
    assert params_list[0]["weight_decay"] == 0
    assert len(params_list[1]["params"]) == num_weight_decay
    assert params_list[1]["weight_decay"] == wd


@pytest.mark.parametrize("shape", [[], [3, 32, 32]])
@pytest.mark.parametrize(
    "batch_size, num_data", [(5, 3), (128, 3), (50, 2), (3, 168), (168, 168)]
)
def test_interleave(shape, batch_size, num_data):
    print(shape, batch_size, num_data)
    inputs = [torch.randint(100, [batch_size] + shape) for _ in range(num_data)]
    orig_inputs = deepcopy(inputs)

    # test_interleave

    o_1 = interleave(inputs, batch_size)
    o_3 = my_interleave(inputs, batch_size)

    for l, r in zip(o_1, o_3):
        assert torch.all(torch.eq(l, r))

    # reverse interleave
    o_1 = interleave(o_1, batch_size)
    o_3 = my_interleave(o_3, batch_size)

    for a_, l, r in zip(orig_inputs, o_1, o_3):
        assert torch.all(torch.eq(a_, r))
        assert torch.all(torch.eq(l, r))
