import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, h_a, h_b):
        g = self.gate(torch.cat([h_a, h_b], dim=-1))
        return g * h_a + (1.0 - g) * h_b
