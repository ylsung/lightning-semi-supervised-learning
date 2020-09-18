import torch
import torch.nn as nn
import torch.nn.functional as F

from .torch_utils import mixup_data, soft_cross_entropy, l2_distribution_loss

def sharpening(label, T):
    label = label.pow(1 / T)
    return label / label.sum(-1, keepdim=True)

class Mixmatch:
    def __init__(self, alpha=0.75, T=0.5, lambda_u=100):
        self.alpha = alpha
        self.T = T
        self.lambda_u = lambda_u

    def loss(self, model, labeled_x, labeled_y, unlabeled_xs):
        """
        labeled_x: [B, :]
        labeled_y: [B, n_classes]
        unlabeled_xs: K unlabeled_x
        """
        
        # size [B, n_classes]
        p_label = self.psuedo_label(model, unlabeled_xs)
        
        B = labeled_x.shape[0]
        K = len(unlabeled_xs)
        # size [(K + 1) * B, :]
        concat_x = torch.cat([labeled_x] + unlabeled_xs, dim=0)
        concat_y = torch.cat(
            [labeled_y, p_label.repeat(K, 1)],
            dim=0
        )

        mixed_x, mixed_y = mixup_data(concat_x, concat_y, self.alpha)

        y_hat = model(mixed_x)

        l_l = soft_cross_entropy(y_hat[:B], concat_y[:B])
        l_u = l2_distribution_loss(y_hat[B:], concat_y[B:])

        return {"loss": l_l + self.lambda_u * l_u,
                "labeled_loss": l_l,
                "unlabeled_loss": l_u}
        
    @torch.no_grad()
    def psuedo_label(self, model, unlabeled_xs):
        "unlabeled_xs is list of unlabeled data"
        p_labels = [F.softmax(model(x), dim=-1) for x in unlabeled_xs]
        # print(p_labels)
        p_label = torch.stack(p_labels).mean(0).detach()
        return sharpening(p_label, self.T)

if __name__ == "__main__":
    from .torch_utils import smooth_label

    n_classes = 4
    a = torch.randn(3, 3)
    y = torch.LongTensor([0, 1, 2])
    y = smooth_label(y, n_classes, 0)
    a_list = [torch.randn(3, 3) for _ in range(6)]
    m = nn.Linear(3, n_classes)

    mixmatch = Mixmatch()

    print(mixmatch.loss(m, a, y, a_list))
