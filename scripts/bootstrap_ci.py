"""Bootstrap 95% confidence intervals for Accuracy, QWK and ECE across every
evaluated configuration, using the saved test-set logits (no retraining
needed). Uses temperature-scaled probabilities to match the numbers already
reported in the paper.

Usage:
    python scripts/bootstrap_ci.py
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

N_BOOT = 1000
SEED = 42

RUNS = [
    ("runs/stage2", "ddr_only", "ddr_only", "DDR-only"),
    ("runs/stage2", "aptos_only", "aptos_only", "APTOS-only"),
    ("runs/stage2", "eyepacs_only", "eyepacs_only", "EyePACS-only"),
    ("runs/stage2", "teacher", "teacher", "Combined (teacher)"),
    ("runs/stage2", "loeo_heldout_eyepacs", "loeo_heldout_eyepacs", "LOEO held-out EyePACS"),
    ("runs/stage2", "loeo_heldout_ddr", "loeo_heldout_ddr", "LOEO held-out DDR"),
    ("runs/stage2", "loeo_heldout_aptos", "loeo_heldout_aptos", "LOEO held-out APTOS"),
    ("runs/ablation_labelfree", "labelfree", "labelfree", "Label-free (Mod. 3)"),
    ("runs/ablation_nocausal", "nocausal", "nocausal", "No-causal (Mod. 1)"),
    ("runs/student", "student", "student", "Distilled student"),
    ("runs/stage2", "deepdrid_teacher", "deepdrid_teacher", "DeepDRiD (external)"),
    ("runs/stage2", "idrid_teacher", "idrid_teacher", "IDRiD (external)"),
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


def ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1); pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0: continue
        e += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def bootstrap_ci(probs, labels, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(labels)
    accs, qwks, eces = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p, y = probs[idx], labels[idx]
        pred = p.argmax(1)
        accs.append((pred == y).mean())
        qwks.append(cohen_kappa_score(y, pred, weights="quadratic"))
        eces.append(ece(p, y))
    def ci(vals):
        vals = np.array(vals)
        return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    return ci(accs), ci(qwks), ci(eces)


def main():
    print(f"{'Configuration':<24}{'Accuracy (95% CI)':<28}{'QWK (95% CI)':<28}{'ECE (95% CI)'}")
    print("-" * 108)
    for base, val_stub, test_stub, name in RUNS:
        val_path = os.path.join(base, f"preds_{val_stub}_val.npz")
        test_path_candidates = glob.glob(os.path.join(base, f"preds_{test_stub}_*.npz"))
        test_path = None
        for p in test_path_candidates:
            if "val" not in p:
                test_path = p
                break
        if not test_path or not os.path.isfile(val_path):
            # external runs have no matching val split for temperature; use test as-is (raw)
            if test_path:
                test = np.load(test_path)
                probs = torch.softmax(torch.from_numpy(test["logits"]).float(), dim=-1).numpy()
                (a, alo, ahi), (q, qlo, qhi), (e, elo, ehi) = bootstrap_ci(probs, test["labels"])
                print(f"{name:<24}{a:.4f} ({alo:.4f}-{ahi:.4f})   {q:.4f} ({qlo:.4f}-{qhi:.4f})   {e:.4f} ({elo:.4f}-{ehi:.4f})")
            else:
                print(f"{name:<24} [skip: missing files]")
            continue

        val = np.load(val_path)
        test = np.load(test_path)
        T = fit_temperature(val["logits"], val["labels"])
        probs = torch.softmax(torch.from_numpy(test["logits"]).float() / T, dim=-1).numpy()
        (a, alo, ahi), (q, qlo, qhi), (e, elo, ehi) = bootstrap_ci(probs, test["labels"])
        print(f"{name:<24}{a:.4f} ({alo:.4f}-{ahi:.4f})   {q:.4f} ({qlo:.4f}-{qhi:.4f})   {e:.4f} ({elo:.4f}-{ehi:.4f})")


if __name__ == "__main__":
    main()
