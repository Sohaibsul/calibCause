"""Run vector-scaling calibration on every saved teacher/per-source run and
report accuracy + ECE (uncalibrated, temperature-scaled, vector-scaled) for
each, so all individual runs can be compared against the combined teacher.

Skips runs whose val split has no ground-truth label variety needed to fit
(handles LOEO/external gracefully) or whose files are missing.

Usage:
    python scripts/vector_scaling_all.py --runs-dir runs/stage2
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

# (val_file_stub, test_file_stub, nice name)
RUNS = [
    ("ddr_only", "ddr_only", "DDR-only"),
    ("aptos_only", "aptos_only", "APTOS-only"),
    ("eyepacs_only", "eyepacs_only", "EyePACS-only"),
    ("teacher", "teacher", "Combined (teacher)"),
    ("loeo_heldout_eyepacs", "loeo_heldout_eyepacs", "LOEO held-out EyePACS"),
    ("loeo_heldout_ddr", "loeo_heldout_ddr", "LOEO held-out DDR"),
    ("loeo_heldout_aptos", "loeo_heldout_aptos", "LOEO held-out APTOS"),
]


def fit_temperature(logits, labels, max_iter=200):
    lg = torch.from_numpy(logits).float()
    lb = torch.from_numpy(labels).long()
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_T.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.detach().exp())


def fit_vector_scale(logits, labels, max_iter=300):
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


def raw_ece(logits, labels):
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    return ece(probs, labels), float((probs.argmax(1) == labels).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/stage2")
    args = ap.parse_args()

    print(f"{'Run':<24}{'Acc(raw)':>10}{'ECE(raw)':>10}{'ECE(T)':>10}{'Acc(vec)':>10}{'ECE(vec)':>10}")
    print("-" * 74)

    for val_stub, test_stub, name in RUNS:
        val_path = os.path.join(args.runs_dir, f"preds_{val_stub}_val.npz")
        test_path = os.path.join(args.runs_dir, f"preds_{test_stub}_test.npz")
        if not (os.path.isfile(val_path) and os.path.isfile(test_path)):
            print(f"{name:<24}  [skip: missing val/test npz]")
            continue

        val = np.load(val_path)
        test = np.load(test_path)

        raw_e, raw_a = raw_ece(test["logits"], test["labels"])

        T = fit_temperature(val["logits"], val["labels"])
        t_probs = torch.softmax(torch.from_numpy(test["logits"]).float() / T, dim=-1).numpy()
        t_e = ece(t_probs, test["labels"])

        w, b = fit_vector_scale(val["logits"], val["labels"])
        v_logits = test["logits"] * w + b
        v_probs = torch.softmax(torch.from_numpy(v_logits).float(), dim=-1).numpy()
        v_e = ece(v_probs, test["labels"])
        v_a = float((v_probs.argmax(1) == test["labels"]).mean())

        print(f"{name:<24}{raw_a:>10.4f}{raw_e:>10.4f}{t_e:>10.4f}{v_a:>10.4f}{v_e:>10.4f}")


if __name__ == "__main__":
    main()
