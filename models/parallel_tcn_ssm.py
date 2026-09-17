import torch
import torch.nn as nn
from models.tcn_backbone import TemporalConvNet
from models.ssm_blocks import ResidualSSMBlockCumsum
from models.fusion_blocks import GatedFusion


class JointSpecificHead(nn.Module):
    def __init__(self, d_model, hidden_dim=64, output_dim=4, dropout=0.0):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.knee_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, h, contact_mask=None):
        z = self.shared(h)
        hip = self.hip_head(z)
        knee = self.knee_head(z)
        return torch.cat([hip, knee], dim=-1)


class ContactAwareHead(nn.Module):
    def __init__(self, d_model, hidden_dim=64, output_dim=4, dropout=0.1):
        super().__init__()

        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        self.swing_head = make_head()
        self.stance_head = make_head()

    def forward(self, h, contact_mask=None):
        pred_swing = self.swing_head(h)
        pred_stance = self.stance_head(h)

        if contact_mask is None:
            return 0.5 * pred_swing + 0.5 * pred_stance

        contact_mask = contact_mask.bool().view(-1, 1)
        pred = torch.where(contact_mask, pred_stance, pred_swing)
        return pred


class ParallelTCNSSMLast(nn.Module):
    def __init__(self, input_dim, output_dim=4, num_channels=None,
                 kernel_size=7, dropout=0.4,
                 ssm_branch_layers=1, ssm_dropout=0.1, ssm_scale_init=0.1,
                 ssm_expand=2, ssm_d_state=8,
                 fusion='gated',
                 head_type='linear', head_hidden_dim=64, head_dropout=0.0,
                 use_angle_only_swing=False, angle_feature_dim=12):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64]
        d_model = num_channels[-1]

        self.head_type = head_type
        self.use_angle_only_swing = use_angle_only_swing
        self.angle_feature_dim = angle_feature_dim

        self.tcn_branch = TemporalConvNet(input_dim, num_channels,
                                          kernel_size=kernel_size, dropout=dropout)

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

        self.tcn_proj = nn.Linear(input_dim, d_model)

        if use_angle_only_swing:
            self.swing_tcn_branch = TemporalConvNet(
                angle_feature_dim, num_channels,
                kernel_size=kernel_size, dropout=dropout
            )
            self.swing_ssm_branch = nn.ModuleList([
                ResidualSSMBlockCumsum(
                    d_model=d_model,
                    d_state=ssm_d_state,
                    expand=ssm_expand,
                    dropout=ssm_dropout,
                    ssm_scale_init=ssm_scale_init
                )
                for _ in range(ssm_branch_layers)
            ])
            self.swing_tcn_proj = nn.Linear(angle_feature_dim, d_model)
            if fusion == 'gated':
                self.swing_fusion = GatedFusion(d_model)
            else:
                self.swing_fusion = None

        if fusion == 'gated':
            self.fusion = GatedFusion(d_model)
        else:
            self.fusion = None

        if head_type == 'joint_specific':
            self.head = JointSpecificHead(d_model, hidden_dim=head_hidden_dim,
                                          output_dim=output_dim, dropout=head_dropout)
        elif head_type == 'contact_aware':
            self.head = ContactAwareHead(d_model, hidden_dim=head_hidden_dim,
                                         output_dim=output_dim, dropout=head_dropout)
        else:
            self.head = nn.Linear(d_model, output_dim)

    def _encode_stance(self, x):
        h_tcn = self.tcn_branch(x.transpose(1, 2)).transpose(1, 2)
        h_ssm_input = self.tcn_proj(x)
        h_ssm = h_ssm_input
        for ssm in self.ssm_branch:
            h_ssm = ssm(h_ssm)
        h_tcn_last = h_tcn[:, -1, :]
        h_ssm_last = h_ssm[:, -1, :]
        if self.fusion is not None:
            h_final = self.fusion(h_tcn_last, h_ssm_last)
        else:
            h_final = (h_tcn_last + h_ssm_last) / 2.0
        return h_final

    def _encode_swing(self, x):
        x_angle = x[:, :, :self.angle_feature_dim]
        h_tcn = self.swing_tcn_branch(x_angle.transpose(1, 2)).transpose(1, 2)
        h_ssm_input = self.swing_tcn_proj(x_angle)
        h_ssm = h_ssm_input
        for ssm in self.swing_ssm_branch:
            h_ssm = ssm(h_ssm)
        h_tcn_last = h_tcn[:, -1, :]
        h_ssm_last = h_ssm[:, -1, :]
        if self.swing_fusion is not None:
            h_final = self.swing_fusion(h_tcn_last, h_ssm_last)
        else:
            h_final = (h_tcn_last + h_ssm_last) / 2.0
        return h_final

    def forward(self, x, contact_mask=None):
        if self.use_angle_only_swing and self.head_type == 'contact_aware':
            h_stance = self._encode_stance(x)
            h_swing = self._encode_swing(x)
            pred_swing = self.head.swing_head(h_swing)
            pred_stance = self.head.stance_head(h_stance)
            if contact_mask is None:
                return 0.5 * pred_swing + 0.5 * pred_stance
            contact_mask = contact_mask.bool().view(-1, 1)
            return torch.where(contact_mask, pred_stance, pred_swing)

        h_final = self._encode_stance(x)

        if self.head_type == 'contact_aware':
            return self.head(h_final, contact_mask=contact_mask)

        return self.head(h_final)
