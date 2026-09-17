import torch
import torch.nn as nn


class DifferenceFeatureBlock(nn.Module):
    def __init__(self, use_first_diff=True, use_second_diff=True):
        super().__init__()
        self.use_first_diff = use_first_diff
        self.use_second_diff = use_second_diff

    def forward(self, angle):
        features = [angle]

        if self.use_first_diff:
            dx = torch.zeros_like(angle)
            dx[:, 1:, :] = angle[:, 1:, :] - angle[:, :-1, :]
            features.append(dx)

        if self.use_second_diff:
            if not self.use_first_diff:
                dx = torch.zeros_like(angle)
                dx[:, 1:, :] = angle[:, 1:, :] - angle[:, :-1, :]

            ddx = torch.zeros_like(angle)
            ddx[:, 1:, :] = dx[:, 1:, :] - dx[:, :-1, :]
            features.append(ddx)

        return torch.cat(features, dim=-1)

    @staticmethod
    def compute_input_dim(base_dim, use_first_diff, use_second_diff):
        dim = base_dim
        if use_first_diff:
            dim += base_dim
        if use_second_diff:
            dim += base_dim
        return dim
