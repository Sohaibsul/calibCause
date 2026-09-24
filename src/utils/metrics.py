"""Accuracy / QWK / AUC / ECE + referable-DR clinical metrics + temperature
scaling (calibration axis).

Extended (QGDR-style) so every report also carries the binary referable-DR
operating-point metrics (sensitivity, specificity, precision, F1) that the base
paper tabulates, plus per-class sensitivity/specificity and balanced accuracy.
All additions are new keys; existing keys are unchanged, so downstream code and
older reports stay compatible.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15):
    """Standard confidence-binned ECE + the data needed for a reliability plot."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "acc": np.nan, "conf": np.nan})
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc_b - conf_b)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "acc": float(acc_b), "conf": float(conf_b)})
    return float(ece), rows


def _binary_operating_point(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sensitivity/specificity/precision/F1 for a 0/1 task from its 2x2 matrix.

    Returns NaNs gracefully if only one class is present (e.g. a cross-env split
    with no referable cases).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall / TPR
    spec = tn / (tn + fp) if (tn + fp) else float("nan")   # TNR
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if (prec + sens) and not np.isnan(prec) and not np.isnan(sens)
          else float("nan"))
    return {
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _per_class_sens_spec(labels: np.ndarray, pred: np.ndarray,
                         n_classes: int) -> dict:
    """One-vs-rest sensitivity and specificity for every grade."""
    per = {}
    for c in range(n_classes):
        yt = (labels == c).astype(int)
        yp = (pred == c).astype(int)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        per[str(c)] = {
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "support": int((labels == c).sum()),
        }
    return per


def grading_metrics(logits: np.ndarray, labels: np.ndarray, n_classes: int = 5) -> dict:
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    pred = probs.argmax(1)
    out = {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "qwk": quadratic_weighted_kappa(labels, pred),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, pred, average="macro",
                                                 zero_division=0)),
        "macro_sensitivity": float(recall_score(labels, pred, average="macro",
                                                zero_division=0)),
    }

    # 5-class one-vs-rest AUC
    try:
        present = np.unique(labels)
        if len(present) == n_classes:
            out["auc_ovr"] = float(roc_auc_score(labels, probs, multi_class="ovr",
                                                 average="macro"))
        else:  # some cross-environment splits do not contain every grade
            out["auc_ovr"] = float("nan")
    except ValueError:
        out["auc_ovr"] = float("nan")

    # referable DR (grade >= 2) -- the clinically actionable binary endpoint
    ref_true = (labels >= 2).astype(int)
    ref_pred = (pred >= 2).astype(int)
    ref_score = probs[:, 2:].sum(1)
    if len(np.unique(ref_true)) == 2:
        out["referable_auc"] = float(roc_auc_score(ref_true, ref_score))
    else:
        out["referable_auc"] = float("nan")
    # sensitivity / specificity / precision / F1 at the argmax operating point
    ref = _binary_operating_point(ref_true, ref_pred)
    out["referable_sensitivity"] = ref["sensitivity"]
    out["referable_specificity"] = ref["specificity"]
    out["referable_precision"] = ref["precision"]
    out["referable_f1"] = ref["f1"]
    out["referable_counts"] = {k: ref[k] for k in ("tp", "fp", "tn", "fn")}

    # per-class sensitivity / specificity (one-vs-rest)
    out["per_class"] = _per_class_sens_spec(labels, pred, n_classes)

    ece, _ = expected_calibration_error(probs, labels)
    out["ece"] = ece
    out["confusion_matrix"] = confusion_matrix(labels, pred,
                                               labels=list(range(n_classes))).tolist()
    return out


class TemperatureScaler(nn.Module):
    """Post-hoc calibration fitted on the validation split (contribution 4).

    Reported alongside raw ECE so the paper can state both the uncalibrated and
    the temperature-scaled reliability of the teacher and the student.
    """

    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    @property
    def T(self) -> float:
        return float(self.log_T.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_T.exp()

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200):
        lg = torch.from_numpy(logits).float()
        lb = torch.from_numpy(labels).long()
        opt = torch.optim.LBFGS([self.log_T], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self(lg), lb)
            loss.backward()
            return loss

        opt.step(closure)
        return self
