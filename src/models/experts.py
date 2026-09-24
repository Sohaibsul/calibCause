"""Stage 2a -- scale-specialised experts and the quality gate."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertHead(nn.Module):
    """One scale-specialised expert.

    fine    -> microaneurysms / small haemorrhages   (stride  8 features)
    medium  -> vascular abnormalities, IRMA, beading (stride 16 features)
    coarse  -> global severe-DR appearance, NVD/NVE  (stride 32 features)
    """

    def __init__(self, dim: int, n_classes: int = 5, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.head(f)


class QualityGate(nn.Module):
    """MLP + softmax -> [g1, g2, g3]  conditioned on image quality.

    Input is the concatenation of the pooled multi-scale context and the scalar
    quality signal (calibrated score in Fix 1, raw SSL score in Module 3).  A
    Fourier-style expansion of the scalar is used so the gate can express a
    non-monotone response to quality instead of a single linear tilt.
    """

    def __init__(self, ctx_dim: int, n_experts: int = 3, hidden: int = 256,
                 n_freq: int = 4, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.n_freq = n_freq
        self.temperature = temperature
        q_dim = 1 + 2 * n_freq
        self.net = nn.Sequential(
            nn.Linear(ctx_dim + q_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_experts))

    def _expand(self, q: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(-1)
        freqs = torch.arange(1, self.n_freq + 1, device=q.device, dtype=q.dtype)
        ang = q * freqs * torch.pi
        return torch.cat([q, torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, ctx: torch.Tensor, q: torch.Tensor):
        z = torch.cat([ctx, self._expand(q)], dim=-1)
        logits = self.net(z) / self.temperature
        return F.softmax(logits, dim=-1), logits
