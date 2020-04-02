import torch
import torch.nn as nn
import torch.nn.functional as F

def sharpening(label, T):
    label = label.pow(1 / T)
    return label / label.sum(-1, keepdim=True)

class Mixmatch:
    def __init__(self, alpha=0.75, T=0.5, lambda_u=100):
        self.alpha = 0.75
        self.T = T
        self.lambda_u = lambda_u

    def loss(self, model, labeled_x, labeled_y, unlabeled_xs):
        pass
    
    def psuedo_label(self, model, unlabeled_xs):
        with torch.no_grad():
            p_labels = [F.softmax(model(x), dim=-1) for x in unlabeled_xs]
        p_label = torch.stack(p_labels).mean(0).detach()
        return sharpening(p_label, self.T)

if __name__ == "__main__":
    a_list = [torch.randn(3, 3) for _ in range(3)]
    m = nn.Linear(3, 4)

    mixmatch = Mixmatch()

    print(mixmatch.psuedo_label(m, a_list))