import torch
import torch.nn as nn
import numpy as np

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

# def half_mixup_data(x, y, alpha):
#     """
#     This function is similar to normal mixup except that the mixed_x 
#     and mixed_y are close to x and y.
#     """
#     # code is modified from 
#     # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

#     batch_size = x.size()[0]

#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)

#     else:
#         lam = 1
    
#     lam = max(lam, 1-lam)

#     index = torch.randperm(batch_size).to(x.device)

#     mixed_x = lam * x + (1 - lam) * x[index]
#     mixed_y = lam * y + (1 - lam) * y[index]

#     return mixed_x, mixed_y 

def smooth_label(y, n_classes, smoothing=0.0):
    """
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

def customized_weight_decay(model, weight_decay, ignore_key=["bn", "bias"]):
    for name, p in model.named_parameters():
        ignore = False
        for key in ignore_key:
            if key in name:
                ignore = True
                break
        if ignore:
            continue
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


def split_weight_decay_weights(model, weight_decay, ignore_key=["bn", "bias"]):
    weight_decay_weights = []
    no_weight_decay_weights = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ignore = False
        for key in ignore_key:
            if key in name:
                ignore = True
                break
        if ignore:
            no_weight_decay_weights.append(p)
        else:
            print(name)
            weight_decay_weights.append(p)

    return [
        {'params': no_weight_decay_weights, 'weight_decay': 0.},
        {'params': weight_decay_weights, 'weight_decay': weight_decay}]

class WeightDecayModule():
    def __init__(self, model, weight_decay, ignore_key=["bn", "bias"]):
        self.weight_decay = weight_decay
        self.available_parameters = []
        for name, p in model.named_parameters():
            ignore = False
            for key in ignore_key:
                if key in name:
                    ignore = True
                    break
            if ignore:
                continue
            print(name)
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
