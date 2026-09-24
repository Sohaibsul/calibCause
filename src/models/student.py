"""Module 2 -- distilled single-stream student (no MoE, no gate)."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SingleStreamStudent(nn.Module):
    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True,
                 n_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.body.num_features, n_classes))

    def forward(self, x: torch.Tensor):
        f = self.body(x)
        return {"logits": self.head(f), "features": f}

    @torch.no_grad()
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
