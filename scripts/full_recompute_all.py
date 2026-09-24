"""Full extended-metrics recomputation (Acc, QWK, macro-F1, AUC_ovr,
referable AUC/Sens/Spec/Prec/F1, ECE) for ALL 10 runs, under both
temperature scaling (matches the numbers already in the paper) and vector
scaling (the improved-calibration candidate), so both can be compared
side by side before deciding which to report.

Usage:
    python scripts/full_recompute_all.py
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)

RUNS = [
    ("runs/stage2", "ddr_only", "ddr_only", "DDR-only"),
    ("runs/stage2", "aptos_only", "aptos_only", "APTOS-only"),
    ("runs/stage2", "eyepacs_only", "eyepacs_only", "EyePACS-only"),
    ("runs/stage2", "teacher", "teacher", "Combined (teacher)"),
    ("runs/stage2", "loeo_heldout_eyepacs", "loeo_heldout_eyepacs", "LOEO held-out EyePACS"),
    ("runs/stage2", "loeo_heldout_ddr", "loeo_heldout_ddr", "LOEO held-out DDR"),
    ("runs/stage2", "loeo_heldout_aptos", "loeo_heldout_aptos", "LOEO held-out APTOS"),
    ("runs/ablation_labelfree", "labelfree", "labelfree", "Ablation: label-free"),
    ("runs/ablation_nocausal", "nocausal", "nocausal", "Ablation: no-causal"),
    ("runs/student", "student", "student", "Distilled student"),
]


def fit_temperature(logits, labels, max_iter=200):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lg / log_T.exp(), lb); loss.backward(); return loss
    opt.step(closure)
    return float(log_T.detach().exp())


def fit_vector_scale(logits, labels, max_iter=300):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    n = logits.shape[1]
    w = torch.ones(n, requires_grad=True); b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.01, max_iter=max_iter)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(lg * w + b, lb); loss.backward(); return loss
    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0: continue
        e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def full_metrics(probs, labels, n_classes=5):
    pred = probs.argmax(1)
    acc = float((pred == labels).mean())
    qwk = float(cohen_kappa_score(labels, pred, weights="quadratic"))
    f1 = float(f1_score(labels, pred, average="macro", zero_division=0))
    try:
        present = np.unique(labels)
        auc_ovr = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")) if len(present) == n_classes else float("nan")
    except ValueError:
        auc_ovr = float("nan")
    ref_true = (labels >= 2).astype(int); ref_pred = (pred >= 2).astype(int)
    ref_score = probs[:, 2:].sum(1)
    ref_auc = float(roc_auc_score(ref_true, ref_score)) if len(np.unique(ref_true)) == 2 else float("nan")
    cm = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    ref_f1 = (2*prec*sens/(prec+sens)) if (prec+sens) and not np.isnan(prec) and not np.isnan(sens) else float("nan")
    e = ece(probs, labels)
    return dict(acc=acc, qwk=qwk, f1=f1, auc_ovr=auc_ovr, ref_auc=ref_auc,
                sens=sens, spec=spec, prec=prec, ref_f1=ref_f1, ece=e)


def fmt(d):
    return (f"acc={d['acc']:.4f} qwk={d['qwk']:.4f} f1={d['f1']:.4f} "
            f"aucOVR={d['auc_ovr']:.4f} aucREF={d['ref_auc']:.4f} "
            f"sens={d['sens']:.4f} spec={d['spec']:.4f} prec={d['prec']:.4f} "
            f"refF1={d['ref_f1']:.4f} ece={d['ece']:.4f}")


def main():
    for base, val_stub, test_stub, name in RUNS:
        val_path = os.path.join(base, f"preds_{val_stub}_val.npz")
        test_path = os.path.join(base, f"preds_{test_stub}_test.npz")
        if not (os.path.isfile(val_path) and os.path.isfile(test_path)):
            print(f"=== {name} ===  [skip: missing npz]\n")
            continue
        val = np.load(val_path); test = np.load(test_path)

        T = fit_temperature(val["logits"], val["labels"])
        t_probs = torch.softmax(torch.from_numpy(test["logits"]).float() / T, dim=-1).numpy()
        t_m = full_metrics(t_probs, test["labels"])

        w, b = fit_vector_scale(val["logits"], val["labels"])
        v_logits = test["logits"] * w + b
        v_probs = torch.softmax(torch.from_numpy(v_logits).float(), dim=-1).numpy()
        v_m = full_metrics(v_probs, test["labels"])

        print(f"=== {name} ===")
        print("  [T]   " + fmt(t_m))
        print("  [vec] " + fmt(v_m))
        print()


if __name__ == "__main__":
    main()
