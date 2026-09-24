"""Objectives for all three training stages."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalAwareCE(nn.Module):
    """Cross-entropy + a soft quadratic-distance penalty.

    DR grading is ordinal: confusing grade 0 with grade 4 is much worse than
    confusing 3 with 4, and QWK -- the metric the base paper reports -- punishes
    exactly that.  Plain CE is blind to it, so a QWK-surrogate term is added.
    """

    def __init__(self, n_classes: int = 5, smoothing: float = 0.05,
                 ordinal_weight: float = 0.5, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.n_classes = n_classes
        self.ordinal_weight = ordinal_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
        grades = torch.arange(n_classes, dtype=torch.float32)
        self.register_buffer("grades", grades)
        self.register_buffer("dist", (grades[None, :] - grades[:, None]) ** 2 /
                             (n_classes - 1) ** 2)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, target)
        if self.ordinal_weight > 0:
            p = F.softmax(logits, dim=-1)
            loss = loss + self.ordinal_weight * (p * self.dist[target]).sum(-1).mean()
        return loss


def quality_regression_loss(q_logit: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """Stage 1 severity regression (BCE on the logit is stabler than MSE on the
    sigmoid because the target is already bounded in [0, 1])."""
    return F.binary_cross_entropy_with_logits(q_logit, q_target)


def pairwise_ranking_loss(q1: torch.Tensor, q2: torch.Tensor,
                          t1: torch.Tensor, t2: torch.Tensor,
                          margin: float = 0.05) -> torch.Tensor:
    """Stage 1 auxiliary: only the *ordering* of two degraded views of the same
    image is supervised, which is what makes the score transfer to real (not
    synthetically degraded) images."""
    sign = torch.sign(t1 - t2)
    return F.relu(margin - sign * (q1 - q2)).mean()


def entropy_regularizer(gate_w: torch.Tensor, target_entropy: float = 0.6) -> torch.Tensor:
    """Keeps the gate from collapsing onto a single expert (a failure mode that
    would make the routing audit in Module 2 meaningless)."""
    ent = -(gate_w.clamp_min(1e-8).log() * gate_w).sum(-1).mean()
    return F.relu(torch.tensor(target_entropy, device=gate_w.device) - ent)


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      target: torch.Tensor, T: float = 6.0, alpha: float = 0.7,
                      hard_loss_fn=None) -> dict:
    """Module 2: student matches teacher soft logits at temperature T (=6)."""
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    F.log_softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean", log_target=True) * (T * T)
    hard = hard_loss_fn(student_logits, target) if hard_loss_fn is not None \
        else F.cross_entropy(student_logits, target)
    return {"loss": alpha * soft + (1 - alpha) * hard, "soft": soft.detach(),
            "hard": hard.detach()}
