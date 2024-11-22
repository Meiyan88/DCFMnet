import torch
from torch import nn

class Class_linear(nn.Module):
    def __init__(self, dim=500, conf=None):
        super(Class_linear, self).__init__()

        self.conf = conf
        self.finally_layer = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        out = self.finally_layer(x)

        return torch.sigmoid(out)