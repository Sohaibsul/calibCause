"""Compute the uncalibrated (raw) ECE for the three runs that were skipped
earlier: label-free, no-causal, and the distilled student.

Usage:
    python scripts/raw_ece_remaining.py
"""
import numpy as np
import torch

RUNS = [
    ("runs/ablation_labelfree/preds_labelfree_test.npz", "Label-free (Mod. 3)"),
    ("runs/ablation_nocausal/preds_nocausal_test.npz", "No-causal (Mod. 1)"),
    ("runs/student/preds_student_test.npz", "Distilled student"),
]

def ece(probs, labels, n_bins=15):
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

for path, name in RUNS:
    d = np.load(path)
    probs = torch.softmax(torch.from_numpy(d["logits"]).float(), dim=-1).numpy()
    e = ece(probs, d["labels"])
    print(f"{name}: raw (uncalibrated) ECE = {e:.4f}")
