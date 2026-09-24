"""Stage 1 (self-supervised quality encoder) + Fix 1 (calibrated quality head).

Figure 1 mapping
----------------
  QualityEncoder          -> "Fundus Quality Encoder (ConvNeXt-Tiny, severity
                              regression q_raw)"                [Stage 1]
  CalibratedQualityHead   -> "Calibrated Quality Head, quality_score in [0,1]"
                              trained on real EyeQ labels        [Fix 1]
  QualityBranch.forward(mode='labelfree') -> "Raw self-supervised score used
                              directly, no human labels"        [Module 3]
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityEncoder(nn.Module):
    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True,
                 embed_dim: int = 256):
        super().__init__()
        self.body = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feat_dim = self.body.num_features
        self.proj = nn.Sequential(nn.Linear(feat_dim, embed_dim), nn.GELU())
        self.q_head = nn.Linear(embed_dim, 1)  # logit -> sigmoid -> q_raw in [0,1]
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        emb = self.proj(self.body(x))
        q_logit = self.q_head(emb).squeeze(-1)
        return {"emb": emb, "q_logit": q_logit, "q_raw": torch.sigmoid(q_logit)}


class CalibratedQualityHead(nn.Module):
    """Maps the frozen SSL embedding to the human EyeQ scale.

    quality_score = E[grade] under the predicted distribution over
    {reject: 0.0, usable: 0.5, good: 1.0}, which keeps it continuous in [0,1]
    and therefore drop-in interchangeable with q_raw for the gate.
    """

    ANCHORS = (0.0, 0.5, 1.0)

    def __init__(self, embed_dim: int = 256, hidden: int = 128, n_classes: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, n_classes))
        self.register_buffer("anchors", torch.tensor(self.ANCHORS))

    def forward(self, emb: torch.Tensor):
        logits = self.mlp(emb)
        score = (F.softmax(logits, dim=-1) * self.anchors).sum(-1)
        return {"quality_logits": logits, "quality_score": score}


class QualityBranch(nn.Module):
    """Wraps the encoder + calibration head and exposes the gating scalar.

    mode = 'calibrated' -> Fix 1   (label-dependent branch)
    mode = 'labelfree'  -> Module 3 (ablation: q_raw used directly)
    The accuracy delta between the two runs is exactly the number Figure 1 calls
    'cost of removing label dependency'.
    """

    def __init__(self, encoder: QualityEncoder, head: CalibratedQualityHead | None = None,
                 mode: str = "calibrated", freeze_encoder: bool = True):
        super().__init__()
        assert mode in {"calibrated", "labelfree"}
        self.encoder = encoder
        self.head = head
        self.mode = mode
        self.frozen = freeze_encoder
        if freeze_encoder:
            # the whole branch is frozen during Stage 2: q is a *given* input to
            # the gate, not something the grading loss is allowed to reshape --
            # otherwise "calibrated quality" would silently drift into a second
            # disease feature and Module 3's comparison would be meaningless.
            for p in self.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, flag: bool = True):
        super().train(flag)
        if self.frozen:  # keep stochastic depth / norm statistics fixed
            self.encoder.eval()
            if self.head is not None:
                self.head.eval()
        return self

    def _compute(self, x: torch.Tensor):
        out = self.encoder(x)
        if self.mode == "labelfree" or self.head is None:
            out["quality_score"] = out["q_raw"]
            out["quality_logits"] = None
        else:
            out.update(self.head(out["emb"]))
        return out

    def forward(self, x: torch.Tensor):
        if self.frozen:
            with torch.no_grad():
                return self._compute(x)
        return self._compute(x)
