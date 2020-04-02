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

def mixup_data(x, y, alpha):
    """
    Args:
        x: data, whose size is [Batch, ...]
        y: label, whose size is [Batch, ...]
        alpha: the paramters for beta distribution. If alpha <= 0 means no mix  
    Return
        mixed inputs, mixed targets
    """
    # codes is borrowed from 
    # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y 

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

def customized_weight_decay(model, weight_decay):
    for p in model.parameters():
        p.data.mul_(1 - weight_decay)

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
        for i in range(len(self.ema_params)):
            self.ema_params[i] = self.ema_params[i].float() 

    def step(self):
        # average all the paramters, including the running mean and 
        # running std in batchnormalization
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.decay)
            ema_param.add_(param * (1 - self.decay))

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
