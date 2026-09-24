"""Module 1 -- Environment-invariant causal block (GRL + discriminator)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class GradientReversalLayer(nn.Module):
    """Identity forward, sign-flipped (and scaled) gradient backward."""

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return grad_reverse(x, self.lambd)


def dann_lambda(step: int, total_steps: int, max_lambda: float = 1.0,
                gamma: float = 10.0) -> float:
    """Ganin's schedule: ramp lambda 0 -> max so the discriminator does not
    destroy the representation before it carries any disease signal."""
    p = min(max(step / max(1, total_steps), 0.0), 1.0)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0))


class CausalTransform(nn.Module):
    """Residual MLP  f -> f'  ('Causal Transform' box in Figure 1).

    Residual so the invariance objective can only *subtract* environment
    information rather than having to re-learn the disease features.
    """

    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.gamma = nn.Parameter(torch.full((dim,), 1e-2))  # near-identity at init

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return f + self.gamma * self.mlp(self.norm(f))


class EnvironmentDiscriminator(nn.Module):
    """Predicts the acquisition environment (EyePACS vs DDR vs APTOS) from f'."""

    def __init__(self, dim: int, hidden: int = 256, n_envs: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, n_envs))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)
