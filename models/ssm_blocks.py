import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@torch.jit.script
def _ssm_sequential_scan(dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                         x_proj: torch.Tensor) -> torch.Tensor:
    B_batch = dt.shape[0]
    L = dt.shape[1]
    d_inner = dt.shape[2]
    d_state = A.shape[1]
    h_t = torch.zeros(B_batch, d_inner, d_state, device=dt.device, dtype=dt.dtype)
    h_list = torch.empty(B_batch, L, d_inner, d_state, device=dt.device, dtype=dt.dtype)
    A_bc = A.unsqueeze(0)
    B_bc = B.unsqueeze(0)
    for t in range(L):
        dA_t = torch.exp(dt[:, t, :].unsqueeze(-1) * A_bc)
        dBx_t = B_bc * dt[:, t, :].unsqueeze(-1) * x_proj[:, t, :].unsqueeze(-1)
        h_t = dA_t * h_t + dBx_t
        h_list[:, t, :, :] = h_t
    return h_list


class ResidualSSMBlockCumsum(nn.Module):
    def __init__(self, d_model, d_state=8, expand=2, dt_min=0.001, dt_max=0.1,
                 dropout=0.1, ssm_scale_init=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(d_model * expand)
        self.d_state = d_state

        self.norm = nn.LayerNorm(d_model)
        self.proj_in = nn.Linear(d_model, self.d_inner, bias=False)
        self.proj_out = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.ssm_scale = nn.Parameter(torch.tensor(ssm_scale_init))

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=3, padding=1, groups=self.d_inner, bias=True
        )

        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.full((self.d_inner, d_state), -1.0))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.C = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)
        self.B = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.1)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt)

    def forward(self, x):
        residual = x
        B_batch, L, D = x.shape

        x_norm = self.norm(x)

        x_proj = self.proj_in(x_norm)
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)
        x_proj = x_proj.transpose(1, 2)
        x_proj = F.silu(x_proj)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(self.dt_proj(x_proj))

        h = _ssm_sequential_scan(dt, A, self.B, x_proj)

        y = (h * self.C.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_proj

        output = self.proj_out(y)
        output = self.dropout(output)

        with torch.no_grad():
            self.ssm_scale.clamp_(0.0, 2.0)

        return residual + self.ssm_scale * output
