"""Compute accuracy, QWK, macro-F1 and ECE (raw / temperature / vector scaling)
for every run, including the two external datasets.

External datasets have no validation split, so the calibration parameters are
fitted on the combined teacher's validation logits and applied to their test
logits (same idea as reusing T=0.59 earlier).

Usage:
    python scripts/ece_f1_all.py
Output:
    printed table + reports/ece_f1_all.csv
"""
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, f1_score

S2 = "runs/stage2"
RUNS = [
    # (name, validation logits used for fitting, test logits)
    ("DDR-only",      f"{S2}/preds_ddr_only_val.npz",      f"{S2}/preds_ddr_only_test.npz"),
    ("APTOS-only",    f"{S2}/preds_aptos_only_val.npz",    f"{S2}/preds_aptos_only_test.npz"),
    ("EyePACS-only",  f"{S2}/preds_eyepacs_only_val.npz",  f"{S2}/preds_eyepacs_only_test.npz"),
    ("Combined",      f"{S2}/preds_teacher_val.npz",       f"{S2}/preds_teacher_test.npz"),
    ("LOEO EyePACS",  f"{S2}/preds_loeo_heldout_eyepacs_val.npz", f"{S2}/preds_loeo_heldout_eyepacs_test.npz"),
    ("LOEO DDR",      f"{S2}/preds_loeo_heldout_ddr_val.npz",     f"{S2}/preds_loeo_heldout_ddr_test.npz"),
    ("LOEO APTOS",    f"{S2}/preds_loeo_heldout_aptos_val.npz",   f"{S2}/preds_loeo_heldout_aptos_test.npz"),
    ("Label-free",    "runs/ablation_labelfree/preds_labelfree_val.npz", "runs/ablation_labelfree/preds_labelfree_test.npz"),
    ("No-causal",     "runs/ablation_nocausal/preds_nocausal_val.npz",   "runs/ablation_nocausal/preds_nocausal_test.npz"),
    ("Student",       "runs/student/preds_student_val.npz",              "runs/student/preds_student_test.npz"),
    # external: calibration borrowed from the combined teacher's validation split
    ("DeepDRiD (ext)", f"{S2}/preds_teacher_val.npz", f"{S2}/preds_deepdrid_teacher_deepdrid.npz"),
    ("IDRiD (ext)",    f"{S2}/preds_teacher_val.npz", f"{S2}/preds_idrid_teacher_idrid.npz"),
]


def fit_temperature(logits, labels, max_iter=200):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def fit_vector_scale(logits, labels, max_iter=300):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    n = logits.shape[1]
    w = torch.ones(n, requires_grad=True)
    b = torch.zeros(n, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.01, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg * w + b, lb)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def softmax(x):
    return torch.softmax(torch.from_numpy(x).float(), dim=-1).numpy()


def score(probs, labels):
    pred = probs.argmax(1)
    return (float((pred == labels).mean()),
            float(cohen_kappa_score(labels, pred, weights="quadratic")),
            float(f1_score(labels, pred, average="macro", zero_division=0)))


rows = []
header = ["config", "acc", "qwk", "macro_f1", "ece_raw", "ece_temp", "ece_vector",
          "acc_vec", "qwk_vec", "macro_f1_vec"]
print(f"{'Config':<16}{'Acc':>7}{'QWK':>8}{'mF1':>7}{'ECEraw':>8}{'ECEtemp':>8}{'ECEvec':>8}{'mF1vec':>8}")
print("-" * 70)

for name, val_path, test_path in RUNS:
    if not (os.path.isfile(val_path) and os.path.isfile(test_path)):
        print(f"{name:<16} [skip: missing file]")
        continue
    val, test = np.load(val_path), np.load(test_path)
    y = test["labels"]

    raw = softmax(test["logits"])
    t = fit_temperature(val["logits"], val["labels"])
    temp = softmax(test["logits"] / t)
    w, b = fit_vector_scale(val["logits"], val["labels"])
    vec = softmax(test["logits"] * w + b)

    acc, qwk, f1 = score(temp, y)        # argmax unchanged by temperature scaling
    acc_v, qwk_v, f1_v = score(vec, y)
    e_raw, e_temp, e_vec = ece(raw, y), ece(temp, y), ece(vec, y)

    rows.append([name, acc, qwk, f1, e_raw, e_temp, e_vec, acc_v, qwk_v, f1_v])
    print(f"{name:<16}{acc:>7.4f}{qwk:>8.4f}{f1:>7.4f}{e_raw:>8.4f}{e_temp:>8.4f}{e_vec:>8.4f}{f1_v:>8.4f}")

os.makedirs("reports", exist_ok=True)
with open("reports/ece_f1_all.csv", "w", newline="") as f:
    csv.writer(f).writerows([header] + [[r[0]] + [f"{v:.4f}" for v in r[1:]] for r in rows])
print("\nsaved -> reports/ece_f1_all.csv")