import torch.nn as nn
from lightning_ssl.models.base_model import CustomModel


class SimpleModel(CustomModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.layer_1 = nn.AvgPool2d(kernel_size=2)
        self.layer_2 = nn.Linear(3, num_classes)

    def forward(self, inputs):
        output = self.layer_1(inputs).reshape(inputs.shape[0], -1)
        return self.layer_2(output)
