"""Vector scaling calibration: fit a per-class scale (and optional bias) on
validation logits, instead of a single global temperature. Uses the same
saved teacher logits as temperature scaling -- no retraining needed.

  z_calibrated = z * w + b   (w, b are length-5 vectors, one per grade)

Usage:
    python scripts/vector_scaling.py \
        --val-preds runs/stage2/preds_teacher_val.npz \
        --test-preds runs/stage2/preds_teacher_test.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


def fit_vector_scale(logits: np.ndarray, labels: np.ndarray, max_iter: int = 300):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    n_classes = logits.shape[1]
    w = torch.ones(n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.01, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg * w + b, lb)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-preds", required=True)
    ap.add_argument("--test-preds", required=True)
    args = ap.parse_args()

    val = np.load(args.val_preds)
    test = np.load(args.test_preds)

    w, b = fit_vector_scale(val["logits"], val["labels"])
    print("fitted per-class weights (w):", np.round(w, 4))
    print("fitted per-class bias    (b):", np.round(b, 4))

    scaled_logits = test["logits"] * w + b
    probs = torch.softmax(torch.from_numpy(scaled_logits).float(), dim=-1).numpy()
    acc = float((probs.argmax(1) == test["labels"]).mean())
    e = ece(probs, test["labels"])

    print(f"\nVector-scaled test accuracy: {acc:.4f}")
    print(f"Vector-scaled test ECE:      {e:.4f}")
    print("(compare: single global T -> 0.1445, per-env T -> 0.1357)")


if __name__ == "__main__":
    main()
