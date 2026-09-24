"""Compute referable-DR clinical metrics (Sens, Spec, Prec, AUC) for the two
external test-only datasets (DeepDRiD, IDRiD), using the already-saved
teacher logits. No val split exists for these (external, test-only), so we
report both raw (uncalibrated) and using the combined-teacher's already
fitted temperature (T=0.590, from the in-domain calibration) for consistency
with the Acc/QWK/ECE numbers already in the paper.

Usage:
    python scripts/external_extended_metrics.py
"""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

RUNS = [
    ("runs/stage2/preds_deepdrid_teacher_deepdrid.npz", "DeepDRiD"),
    ("runs/stage2/preds_idrid_teacher_idrid.npz", "IDRiD"),
]
T_FIXED = 0.590  # combined teacher's temperature, fitted on in-domain validation


def referable_metrics(probs, labels):
    pred = probs.argmax(1)
    ref_true = (labels >= 2).astype(int)
    ref_pred = (pred >= 2).astype(int)
    ref_score = probs[:, 2:].sum(1)

    cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    auc = roc_auc_score(ref_true, ref_score) if len(np.unique(ref_true)) == 2 else float("nan")
    return sens, spec, prec, auc


for path, name in RUNS:
    d = np.load(path)
    logits, labels = d["logits"], d["labels"]

    # raw (uncalibrated)
    raw_probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    sens_r, spec_r, prec_r, auc_r = referable_metrics(raw_probs, labels)

    # using the fixed in-domain temperature (for consistency with reported Acc/QWK/ECE)
    t_probs = torch.softmax(torch.from_numpy(logits).float() / T_FIXED, dim=-1).numpy()
    sens_t, spec_t, prec_t, auc_t = referable_metrics(t_probs, labels)

    print(f"=== {name} ===")
    print(f"  [raw]  sens={sens_r:.4f} spec={spec_r:.4f} prec={prec_r:.4f} auc={auc_r:.4f}")
    print(f"  [T={T_FIXED}] sens={sens_t:.4f} spec={spec_t:.4f} prec={prec_t:.4f} auc={auc_t:.4f}")
    print()
