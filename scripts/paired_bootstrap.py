"""Paired bootstrap for held-out QWK differences between models (reviewer comments 1 and 2).

All models must be evaluated on the SAME held-out test set (same images, same order).
For each bootstrap draw we resample test images with replacement ONCE and score every
model on that same resample, so the difference QWK(A) - QWK(B) is paired and the
resulting confidence interval reflects only test-set sampling noise (it does NOT include
training-seed variance -- every model here is a single seed; say so in the paper).

Usage:
    python scripts/paired_bootstrap.py --fold eyepacs \
        --model full=runs/stage2/preds_loeo_heldout_eyepacs_test.npz \
        --model nocausal=runs/ablation_nocausal/preds_nocausal_loeo_heldout_eyepacs_test.npz \
        --model labelfree=runs/ablation_labelfree/preds_labelfree_loeo_heldout_eyepacs_test.npz \
        --model plain=runs/baseline/preds_baseline2_loeo_heldout_eyepacs_test.npz \
        --out runs/bootstrap_eyepacs.json
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

K = 5
_W = (np.arange(K)[:, None] - np.arange(K)[None, :]) ** 2 / float((K - 1) ** 2)


def qwk(y: np.ndarray, p: np.ndarray) -> float:
    o = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(np.float64)
    n = o.sum()
    e = np.outer(o.sum(1), o.sum(0)) / n
    den = (_W * e).sum()
    return float(1.0 - (_W * o).sum() / den) if den > 0 else 0.0


ENV2ID = {"eyepacs": 0, "ddr": 1, "aptos": 2}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auc_ovr(y: np.ndarray, prob: np.ndarray) -> float:
    """Macro one-vs-rest AUC over the grades present in the resample (same definition as the reports)."""
    from sklearn.metrics import roc_auc_score
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    if len(present) < prob.shape[1]:
        prob = prob[:, present]
        prob = prob / prob.sum(axis=1, keepdims=True)
        return float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=present))
    return float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))


def load(path: str, env: str | None):
    """Labels and argmax predictions of the held-out environment only.

    evaluate.py scores the whole test split of ALL environments; the LOEO number we report is
    the held-out environment's own subset, so the same subset must be selected here."""
    d = np.load(path)
    y = d["labels"].astype(np.int64)
    p = d["logits"].astype(np.float64)
    if env is not None:
        m = d["envs"] == ENV2ID[env]
        y, p = y[m], p[m]
    return y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="held-out environment: eyepacs | ddr | aptos")
    ap.add_argument("--all-envs", action="store_true", help="do not filter to the held-out environment")
    ap.add_argument("--model", action="append", required=True, help="name=path/to/preds_*_test.npz")
    ap.add_argument("--metric", choices=["qwk", "auc"], default="qwk",
                    help="qwk = argmax decision rule; auc = macro one-vs-rest AUC of the softmax probabilities")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names, labels, preds = [], None, {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        y, p = load(path, None if args.all_envs else args.fold)
        if labels is None:
            labels = y
        elif len(y) != len(labels) or not np.array_equal(y, labels):
            raise SystemExit(f"[error] {name}: labels differ from the first model -> not the same test set / "
                             f"order, a paired bootstrap would be invalid.")
        names.append(name)
        preds[name] = p
    n = len(labels)
    if args.metric == "qwk":
        preds = {m: v.argmax(axis=1).astype(np.int64) for m, v in preds.items()}
        score = qwk
    else:
        preds = {m: softmax(v) for m, v in preds.items()}
        score = auc_ovr
    rng = np.random.default_rng(args.seed)

    point = {m: score(labels, preds[m]) for m in names}
    draws = {m: np.empty(args.n_boot) for m in names}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = labels[idx]
        for m in names:
            draws[m][b] = score(yb, preds[m][idx])

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    res = {"fold": args.fold, "n_test": int(n), "n_boot": args.n_boot, "models": {}, "pairs": {}}
    print(f"\nfold={args.fold}  n_test={n}  n_boot={args.n_boot}  ({args.metric})")
    print(f"{'model':<14}{args.metric.upper():>8}   95% CI")
    for m in names:
        lo, hi = ci(draws[m])
        res["models"][m] = {args.metric: point[m], "ci95": [lo, hi]}
        print(f"{m:<14}{point[m]:>8.4f}   [{lo:.4f}, {hi:.4f}]")
    print(f"\n{'A - B':<26}{'delta':>8}   {'95% CI of delta':<20}{'P(delta<=0)':>12}  verdict")
    for a, b in itertools.combinations(names, 2):
        d = draws[a] - draws[b]
        lo, hi = ci(d)
        p_le0 = float((d <= 0).mean())
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (no clear difference)"
        res["pairs"][f"{a}-{b}"] = {"delta": point[a] - point[b], "ci95": [lo, hi], "p_delta_le_0": p_le0}
        print(f"{a + ' - ' + b:<26}{point[a] - point[b]:>8.4f}   [{lo:+.4f}, {hi:+.4f}]      {p_le0:>8.3f}  {verdict}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
