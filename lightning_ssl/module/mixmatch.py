import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning_ssl.utils.torch_utils import (
    half_mixup_data,
    soft_cross_entropy,
    l2_distribution_loss,
    smooth_label,
    customized_weight_decay,
    interleave,
)


class Mixmatch:
    classifier: nn.Module
    hparams: ...
    lambda_u: float

    def _loss(self, labeled_x, labeled_y, unlabeled_xs, batch_inference=False):
        """
        labeled_x: [B, :]
        labeled_y: [B, n_classes]
        unlabeled_xs: K unlabeled_x
        """
        batch_size = labeled_x.size(0)
        num_augmentation = len(unlabeled_xs)  # num_augmentation

        # not to update the running mean and variance in BN
        self.classifier.freeze_running_stats()

        p_unlabeled_y = self.classifier.psuedo_label(
            unlabeled_xs, self.hparams.T, batch_inference
        )

        # # size [(K + 1) * B, :]
        # all_inputs = torch.cat([labeled_x] + unlabeled_xs, dim=0)
        # all_targets = torch.cat(
        #     [labeled_y, p_unlabeled_y.repeat(K, 1)],
        #     dim=0
        # )

        # mixed_input, mixed_target = half_mixup_data(all_inputs, all_targets, self.hparams.alpha)

        # logits = self.forward(mixed_input, self.classifier)

        # l_l = soft_cross_entropy(logits[:batch_size], mixed_target[:batch_size])
        # l_u = l2_distribution_loss(logits[batch_size:], mixed_target[batch_size:])

        all_inputs = torch.cat([labeled_x] + unlabeled_xs, dim=0)
        all_targets = torch.cat(
            [labeled_y, p_unlabeled_y.repeat(num_augmentation, 1)], dim=0
        )

        mixed_input, mixed_target = half_mixup_data(
            all_inputs, all_targets, self.hparams.alpha
        )

        if batch_inference:
            self.classifier.recover_running_stats()
            logits = self.classifier(mixed_input)
        else:
            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits_other = [self.classifier(x) for x in mixed_input[1:]]

            self.classifier.recover_running_stats()

            # only update the BN stats for the first batch
            logits_first = self.classifier(mixed_input[0])

            logits = [logits_first] + logits_other

            # put interleaved samples back
            logits = interleave(logits, batch_size)

            logits = torch.cat(logits, dim=0)

        l_l = soft_cross_entropy(logits[:batch_size], mixed_target[:batch_size])
        l_u = l2_distribution_loss(logits[batch_size:], mixed_target[batch_size:])

        return l_l + self.lambda_u * l_u, l_l, l_u
