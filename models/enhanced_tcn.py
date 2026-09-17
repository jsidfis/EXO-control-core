import torch
import torch.nn as nn
from models.tcn_backbone import TemporalConvNet


class EnhancedTCNLast(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_channels=None,
                 kernel_size=7, dropout=0.4):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        self.tcn = TemporalConvNet(input_dim, num_channels,
                                   kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        return output[:, -1, :]
