import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy

def sharpening(label, T):
    label = label.pow(1 / T)
    return label / label.sum(-1, keepdim=True)

def soft_cross_entropy(input_, target_, dim=-1):
    """
    compute the cross entropy between input_ and target_
    Args
        input_: logits of model's prediction. Size = [Batch, n_classes]
        target_: probability of target distribution. Size = [Batch, n_classes]
    Return
        the entropy between input_ and target_
    """

    input_ = input_.log_softmax(dim=dim)

    return torch.mean(torch.sum(- target_ * input_, dim=dim))

def l2_distribution_loss(input_, target_, dim=-1):
    input_ = input_.softmax(dim=dim)

    return torch.mean((input_ - target_) ** 2)

def mixup_data(x, y, alpha):
    """
    Args:
        x: data, whose size is [Batch, ...]
        y: label, whose size is [Batch, ...]
        alpha: the paramters for beta distribution. If alpha <= 0 means no mix  
    Return
        mixed inputs, mixed targets
    """
    # code is modified from 
    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    
    batch_size = x.size()[0]

    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).rsample((batch_size, ))
    else:
        lam = torch.ones((batch_size, ))
    
    lam = lam.to(x.device)

    index = torch.randperm(batch_size).to(x.device)

    x_size, y_size = [1 for _ in range(len(x.shape))], [1 for _ in range(len(y.shape))]
    x_size[0], y_size[0] = batch_size, batch_size

    mixed_x = lam.view(x_size) * x + (1 - lam.view(x_size)) * x[index]
    mixed_y = lam.view(y_size) * y + (1 - lam.view(y_size)) * y[index]

    return mixed_x, mixed_y 

def mixup_data_for_testing(x, y, alpha):
    """
    Args:
        x: data, whose size is [Batch, ...]
        y: label, whose size is [Batch, ...]
        alpha: the paramters for beta distribution. If alpha <= 0 means no mix  
    Return
        mixed inputs, mixed targets
    """
    # code is modified from 
    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    
    batch_size = x.size()[0]

    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).rsample((batch_size, ))
    else:
        lam = torch.ones((batch_size, ))
    
    lam = lam.to(x.device)

    index = torch.randperm(batch_size).to(x.device)

    x_size, y_size = [1 for _ in range(len(x.shape))], [1 for _ in range(len(y.shape))]
    x_size[0], y_size[0] = batch_size, batch_size

    mixed_x = lam.view(x_size) * x + (1 - lam.view(x_size)) * x[index]
    mixed_y = lam.view(y_size) * y + (1 - lam.view(y_size)) * y[index]

    return mixed_x, mixed_y, x[index], y[index], lam

def half_mixup_data(x, y, alpha):
    """
    This function is similar to normal mixup except that the mixed_x 
    and mixed_y are close to x and y.
    """
    # code is modified from 
    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

    batch_size = x.size()[0]

    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).rsample((batch_size, ))
    else:
        lam = torch.ones((batch_size, ))
    
    lam = torch.max(lam, 1 - lam)

    lam = lam.to(x.device)

    index = torch.randperm(batch_size).to(x.device)

    x_size, y_size = [1 for _ in range(len(x.shape))], [1 for _ in range(len(y.shape))]
    x_size[0], y_size[0] = batch_size, batch_size

    mixed_x = lam.view(x_size) * x + (1 - lam.view(x_size)) * x[index]
    mixed_y = lam.view(y_size) * y + (1 - lam.view(y_size)) * y[index]

    return mixed_x, mixed_y 

def half_mixup_data_for_testing(x, y, alpha):
    """
    This function is similar to normal mixup except that the mixed_x 
    and mixed_y are close to x and y.
    """
    # code is modified from 
    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

    batch_size = x.size()[0]

    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).rsample((batch_size, ))
    else:
        lam = torch.ones((batch_size, ))
    
    lam = torch.max(lam, 1 - lam)

    lam = lam.to(x.device)

    index = torch.randperm(batch_size).to(x.device)

    x_size, y_size = [1 for _ in range(len(x.shape))], [1 for _ in range(len(y.shape))]
    x_size[0], y_size[0] = batch_size, batch_size

    mixed_x = lam.view(x_size) * x + (1 - lam.view(x_size)) * x[index]
    mixed_y = lam.view(y_size) * y + (1 - lam.view(y_size)) * y[index]

    return mixed_x, mixed_y , x[index], y[index], lam

def smooth_label(y, n_classes, smoothing=0.0):
    """
    Transform the y into one-hot representation and smooth it.
    If smoothing is 0, then the return will be one-hot representation of y.
    Args
        y: label which is LongTensor
        n_classes: the total number of classes
        smoothing: the paramter of label smoothing
    Return
        true: the smooth label whose size is [Batch, n_classes] 
    """

    confidence = 1.0 - smoothing
    true_dist = torch.zeros(*list(y.size()), n_classes).to(y.device)
    true_dist.fill_(smoothing / (n_classes - 1))
    true_dist.scatter_(-1, y.data.unsqueeze(1), confidence)

    return true_dist

def to_one_hot_vector(y, n_classes):
    return smooth_label(y, n_classes, 0)

def customized_weight_decay(model, weight_decay, ignore_key=["bn", "bias"]):
    for name, p in model.named_parameters():
        if not any(key in name for key in ignore_key):
            p.data.mul_(1 - weight_decay)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def my_interleave_1(inputs, batch_size):
    """
    * Make the data interleave. 
    * Swap the data of the first batch (inputs[0]) to other batches.
    * change_indices would be a increasing function.
    * len(change_indices) should be the same as len(inputs[0]), and the elements
      denote which row should the data in first row change with.
    """
    ret = deepcopy(inputs)
    inputs_size = len(inputs)

    repeat = batch_size // inputs_size
    residual = batch_size % inputs_size
    change_indices = list(range(inputs_size)) * repeat + list(range(inputs_size - residual, inputs_size))
    change_indices = sorted(change_indices)
    # print(change_indices)
    for i, switch_row in enumerate(change_indices):
        ret[0][i], ret[switch_row%inputs_size][i] = inputs[switch_row%inputs_size][i], inputs[0][i]

    return ret

def my_interleave_2(inputs, batch_size):
    """
    * Make the data interleave. 
    * Swap the data of the first batch (inputs[0]) to other batches.
    * change_indices would be a increasing function.
    * len(change_indices) should be the same as len(inputs[0]), and the elements
      denote which row should the data in first row change with.
    """
    # ret = deepcopy(inputs)
    def swap(A, B):
        return B.clone(), A.clone()

    ret = inputs
    inputs_size = len(inputs)

    # equally switch the first row to other rows, so we compute how many repeat for range(inputs_size),
    # which store the rows to change. 
    # some of the element cannot evenly spread two rows, so we preferentially use the rows which are farer to 0th row.
    repeat = batch_size // inputs_size
    residual = batch_size % inputs_size
    change_indices = list(range(inputs_size)) * repeat + list(range(inputs_size - residual, inputs_size))
    change_indices = sorted(change_indices)
    # print(change_indices)

    # start to change elements
    for i, switch_row in enumerate(change_indices):
        ret[0][i], ret[switch_row%inputs_size][i] = swap(ret[0][i], ret[switch_row%inputs_size][i])

    return ret

def my_interleave(inputs, batch_size):
    """
    * This function will override inputs
    * Make the data interleave. 
    * Swap the data of the first batch (inputs[0]) to other batches.
    * change_indices would be a increasing function.
    * len(change_indices) should be the same as len(inputs[0]), and the elements
      denote which row should the data in first row change with.
    """
    def swap(A, B):
        """
        swap for tensors
        """
        return B.clone(), A.clone()

    ret = inputs
    inputs_size = len(inputs)

    repeat = batch_size // inputs_size
    residual = batch_size % inputs_size

    # equally switch the first row to other rows, so we compute how many repeat for range(inputs_size),
    # which store the rows to change. 
    # some of the element cannot evenly spread two rows, so we preferentially use the rows which are farer to 0th row.

    change_indices = list(range(inputs_size)) * repeat + list(range(inputs_size - residual, inputs_size))
    change_indices = sorted(change_indices)
    # print(change_indices)

    # the change_indices is monotone increasing function, so we can group the same elements and swap together
    # e.g. change_indices = [0, 1, 1, 2, 2, 2]
    #      => two_dimension_change_indices = [[0], [1, 1], [2, 2, 2]] 
    two_dimension_change_indices = []
    
    change_indices.insert(0, -1)
    change_indices.append(change_indices[-1]+1)
    start = 0
    for i in range(1, len(change_indices)):
        if change_indices[i] != change_indices[i-1]:
            two_dimension_change_indices.append(change_indices[start: i])
            start = i

    two_dimension_change_indices.pop(0)
    
    i = 0
    for switch_rows in two_dimension_change_indices:
        switch_row = switch_rows[0]
        num = len(switch_rows)
        ret[0][i:i+num], ret[switch_row%inputs_size][i:i+num] = swap(ret[0][i:i+num], ret[switch_row%inputs_size][i:i+num])
        i += num

    return ret


def split_weight_decay_weights(model, weight_decay, ignore_key=["bn", "bias"]):
    weight_decay_weights = []
    no_weight_decay_weights = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(key in name for key in ignore_key):
            no_weight_decay_weights.append(p)
        else:
            # print(name)
            weight_decay_weights.append(p)

    return [
        {'params': no_weight_decay_weights, 'weight_decay': 0.},
        {'params': weight_decay_weights, 'weight_decay': weight_decay}]

class WeightDecayModule():
    def __init__(self, model, weight_decay, ignore_key=["bn", "bias"]):
        self.weight_decay = weight_decay
        self.available_parameters = []
        for name, p in model.named_parameters():
            if not any(key in name for key in ignore_key):
                # print(name)
                self.available_parameters.append(p)

    def decay(self):
        for p in self.available_parameters:
            p.data.mul_(1 - self.weight_decay)

class EMA():
    """
        The module for exponential moving average
    """
    def __init__(self, model, ema_model, decay=0.999):
        self.decay = decay
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        # some of the quantity in batch norm is LongTensor,
        # we have to make them become float, or it will cause the 
        # type error in mul_ or add_ in self.step()
        for p in ema_model.parameters():
            p.detach_()
        for i in range(len(self.ema_params)):
            self.ema_params[i] = self.ema_params[i].float() 

    def step(self):
        # average all the paramters, including the running mean and 
        # running std in batchnormalization
        for param, ema_param in zip(self.params, self.ema_params):
            # if param.dtype == torch.float32:
            ema_param.mul_(self.decay)
            ema_param.add_(param * (1 - self.decay))
            # if param.dtype == torch.float32:
            #     param.mul_(1 - 4e-5)


class LabelSmoothingLoss(nn.Module):
    """
        The loss module for label smoothing
    """
    # codes borrowed from https://github.com/pytorch/pytorch/issues/7455
    def __init__(self, classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target, dim=-1):
        # the pred should be logits
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred).to(target.device)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(dim, target.data.unsqueeze(1), self.confidence)

        return soft_cross_entropy(pred, true_dist, dim=-1)
