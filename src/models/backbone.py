"""Multi-scale feature extractor (Figure 1, Stage 2a: 'Multi-Scale Backbone')."""
from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


def _to_nchw(t: torch.Tensor, channels: int) -> torch.Tensor:
    """timm returns (B,H,W,C) for Swin/ViT-style nets and (B,C,H,W) for ConvNeXt.

    Disambiguated with the channel count reported by feature_info rather than by
    guessing from the shape, so a square feature map cannot be mis-permuted.
    """
    if t.ndim == 4 and t.shape[1] != channels and t.shape[-1] == channels:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


class MultiScaleBackbone(nn.Module):
    """Returns three feature maps: fine (stride 8), medium (16), coarse (32).

    Default is ConvNeXt-Tiny as drawn in Figure 1; pass `name='swin_tiny_patch4_window7_224'`
    to reproduce the original QGDR backbone instead.
    """

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True,
                 out_indices=(1, 2, 3), drop_path_rate: float = 0.1):
        super().__init__()
        self.body = timm.create_model(
            name, pretrained=pretrained, features_only=True,
            out_indices=out_indices, drop_path_rate=drop_path_rate)
        self.dims: List[int] = list(self.body.feature_info.channels())
        self.scales = ("fine", "medium", "coarse")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [_to_nchw(f, c) for f, c in zip(self.body(x), self.dims)]
