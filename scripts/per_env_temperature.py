"""Fit a separate temperature per environment (instead of one global T) and
report the resulting combined ECE, using the already-saved teacher logits.

Requires runs/stage2/preds_teacher_val.npz and preds_teacher_test.npz, each
expected to contain arrays: logits, labels, env (env as string or int array
matching ENV2ID order: eyepacs=0, ddr=1, aptos=2).

Usage:
    python scripts/per_env_temperature.py \
        --val-preds runs/stage2/preds_teacher_val.npz \
        --test-preds runs/stage2/preds_teacher_test.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F


ENV_NAMES = {0: "eyepacs", 1: "ddr", 2: "aptos"}


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
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

    val_env = val["envs"]
    test_env = test["envs"]
    # normalise env array to ints if it's strings
    def to_ids(arr):
        if arr.dtype.kind in ("U", "S"):
            name2id = {v: k for k, v in ENV_NAMES.items()}
            return np.array([name2id[str(x)] for x in arr])
        return arr.astype(int)

    val_env_ids = to_ids(val_env)
    test_env_ids = to_ids(test_env)

    all_probs = np.zeros((len(test["labels"]), test["logits"].shape[1]))
    temps = {}
    for eid, ename in ENV_NAMES.items():
        vmask = val_env_ids == eid
        tmask = test_env_ids == eid
        if vmask.sum() == 0 or tmask.sum() == 0:
            print(f"  [skip] {ename}: no val or test rows")
            continue
        T = fit_temperature(val["logits"][vmask], val["labels"][vmask])
        temps[ename] = T
        probs = torch.softmax(torch.from_numpy(test["logits"][tmask]).float() / T, dim=-1).numpy()
        all_probs[tmask] = probs
        env_ece = ece(probs, test["labels"][tmask])
        print(f"  {ename}: T={T:.4f}  n_test={tmask.sum()}  ECE={env_ece:.4f}")

    combined_ece = ece(all_probs, test["labels"])
    print(f"\nPer-environment temperatures: {temps}")
    print(f"Combined ECE with per-environment T: {combined_ece:.4f}")
    print("(compare to single-T combined ECE reported earlier: 0.1445)")


if __name__ == "__main__":
    main()
