import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_ssl.utils.torch_utils import sharpening

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.batchnorms, self.momentums = None, None

    @torch.no_grad()
    def psuedo_label(self, unlabeled_xs, temperature, batch_inference=False):
        """
        Generate the pseudo labels for given unlabeled_xs.
        Args
            unlabeled_xs: list of unlabeled data (torch.FloatTensor), 
                          e.g. [unlabeled_data_1, unlabeled_data_2, ..., unlabeled_data_n]
                          Note that i th element in those unlabeled data should has same 
                          semantic meaning (come from different data augmentations).  
            temperature: the temperature parameter in softmax
        """
        if batch_inference:
            batch_size, num_augmentations = unlabeled_xs[0].shape[0], len(unlabeled_xs)
            unlabeled_xs = torch.cat(unlabeled_xs, dim=0)
            p_labels = F.softmax(self(unlabeled_xs), dim=-1)
            p_labels = p_labels.view(num_augmentations, batch_size, -1)
            p_label = p_labels.mean(0).detach()
        else:
            p_labels = [F.softmax(self(x), dim=-1) for x in unlabeled_xs]
            p_label = torch.stack(p_labels).mean(0).detach()

        return sharpening(p_label, temperature)

    def extract_norm_n_momentum(self):
        # Extract the batchnorms and their momentum
        original_momentums = []
        batchnorms = []
        for module in self.modules():
            if any(isinstance(module, b) for b in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]):
                original_momentums.append(module.momentum)
                batchnorms.append(module)
                
        return batchnorms, original_momentums

    def extract_running_stats(self):
        # Extract the running stats of batchnorms
        running_stats = []
        for module in self.modules():
            if any(isinstance(module, b) for b in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]):
                running_stats.append([module.running_mean, module.running_var])

        return running_stats

    def freeze_running_stats(self):
        # Set the batchnorms' momentum to 0 to freeze the running stats
        # First call
        if None in [self.batchnorms, self.momentums]:
            self.batchnorms, self.momentums = self.extract_norm_n_momentum()

        for module in self.batchnorms:
            module.momentum = 0

    def recover_running_stats(self):
        # Recover the batchnorms' momentum to make running stats updatable
        if None in [self.batchnorms, self.momentums]:
            return
        for module, momentum in zip(self.batchnorms, self.momentums):
            module.momentum = momentum

    