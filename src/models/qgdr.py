"""QGDR++ teacher: quality-gated MoE + environment-invariant causal block.

Forward graph (matches Figure 1 top-to-bottom):

    fundus image ->  MultiScaleBackbone  -> [fine, medium, coarse] maps
                              |                         |
                              |                    GAP + pooling
                              |                         v
                              |            CausalTransform (per scale) -> f'
                              |                    /        |        \\
        QualityBranch -> q ---+---> QualityGate  <-'        |         `-> Experts
                                        |                   |               |
                                        |                  GRL              |
                                        |                   v               v
                                        `--- g1,g2,g3 --> EnvDiscriminator  logits_s
                                                                  \\        /
                                                              weighted sum -> DR logits (teacher)
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import MultiScaleBackbone
from .causal import CausalTransform, EnvironmentDiscriminator, grad_reverse
from .experts import ExpertHead, QualityGate
from .quality import QualityBranch


class QGDRPlus(nn.Module):
    def __init__(self,
                 quality_branch: QualityBranch,
                 backbone_name: str = "convnext_tiny",
                 pretrained: bool = True,
                 n_classes: int = 5,
                 n_envs: int = 3,
                 proj_dim: int = 384,
                 use_causal: bool = True,
                 use_env_disc: bool = True,
                 gate_temperature: float = 1.0):
        super().__init__()
        self.quality = quality_branch
        self.backbone = MultiScaleBackbone(backbone_name, pretrained=pretrained)
        self.n_classes = n_classes
        self.use_causal = use_causal
        self.use_env_disc = use_env_disc
        self.scales = self.backbone.scales

        # per-scale pooling + projection to a common width
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, proj_dim, 1), nn.GroupNorm(8, proj_dim), nn.GELU())
            for d in self.backbone.dims])
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Module 1
        self.causal = nn.ModuleList([CausalTransform(proj_dim) for _ in self.backbone.dims]) \
            if use_causal else None
        ctx_dim = proj_dim * len(self.backbone.dims)
        self.env_disc = EnvironmentDiscriminator(ctx_dim, n_envs=n_envs) if use_env_disc else None

        # Stage 2a
        self.experts = nn.ModuleList([ExpertHead(proj_dim, n_classes) for _ in self.backbone.dims])
        self.gate = QualityGate(ctx_dim, n_experts=len(self.backbone.dims),
                                temperature=gate_temperature)

        self._feature_maps: List[torch.Tensor] = []  # kept for Grad-CAM

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0,
                keep_maps: bool = False) -> Dict[str, torch.Tensor]:
        qout = self.quality(x)
        q = qout["quality_score"]

        maps = self.backbone(x)
        maps = [p(m) for p, m in zip(self.proj, maps)]
        if keep_maps:
            self._feature_maps = maps  # non-detached: Grad-CAM needs the graph
        feats = [self.pool(m).flatten(1) for m in maps]

        if self.causal is not None:
            feats = [c(f) for c, f in zip(self.causal, feats)]

        ctx = torch.cat(feats, dim=-1)
        gate_w, gate_logits = self.gate(ctx, q)

        expert_logits = torch.stack([e(f) for e, f in zip(self.experts, feats)], dim=1)
        logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)

        out = {
            "logits": logits,
            "expert_logits": expert_logits,   # (B, 3, C)
            "gate_weights": gate_w,           # (B, 3)
            "gate_logits": gate_logits,
            "quality_score": q,
            "quality_logits": qout.get("quality_logits"),
            "q_raw": qout["q_raw"],
            "features": ctx,
        }
        if self.env_disc is not None:
            out["env_logits"] = self.env_disc(grad_reverse(ctx, grl_lambda))
        return out

    # ------------------------------------------------------------------ #
    def trainable_groups(self, lr: float, lr_backbone_mult: float = 0.1):
        """Lower LR for the pretrained backbone, full LR for the new heads."""
        bb, new = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (bb if n.startswith("backbone.") else new).append(p)
        return [{"params": bb, "lr": lr * lr_backbone_mult},
                {"params": new, "lr": lr}]
