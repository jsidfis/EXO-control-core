import torch
import torch.nn as nn
from models.tcn_backbone import TemporalConvNet
from models.ssm_blocks import ResidualSSMBlockCumsum


class TCNOnlyLast(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_channels=None,
                 kernel_size=7, dropout=0.4,
                 head_type='linear', head_hidden_dim=64, head_dropout=0.0):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        d_model = num_channels[-1]

        self.tcn_branch = TemporalConvNet(input_dim, num_channels,
                                          kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x, contact_mask=None):
        h_tcn = self.tcn_branch(x.transpose(1, 2)).transpose(1, 2)
        h_tcn_last = h_tcn[:, -1, :]
        return self.head(h_tcn_last)


class SSMOnlyLast(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_channels=None,
                 kernel_size=7, dropout=0.4,
                 ssm_branch_layers=1, ssm_dropout=0.1, ssm_scale_init=0.1,
                 ssm_expand=2, ssm_d_state=8,
                 head_type='linear', head_hidden_dim=64, head_dropout=0.0):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        d_model = num_channels[-1]

        self.tcn_proj = nn.Linear(input_dim, d_model)

        self.ssm_branch = nn.ModuleList([
            ResidualSSMBlockCumsum(
                d_model=d_model,
                d_state=ssm_d_state,
                expand=ssm_expand,
                dropout=ssm_dropout,
                ssm_scale_init=ssm_scale_init
            )
            for _ in range(ssm_branch_layers)
        ])

        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x, contact_mask=None):
        h_ssm_input = self.tcn_proj(x)
        h_ssm = h_ssm_input
        for ssm in self.ssm_branch:
            h_ssm = ssm(h_ssm)
        h_ssm_last = h_ssm[:, -1, :]
        return self.head(h_ssm_last)
