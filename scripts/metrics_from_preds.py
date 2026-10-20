"""Held-out (unseen-site) metrics recomputed from the saved test predictions, for every LOEO model,
with ONE identical protocol: raw logits, softmax at T = 1, no temperature scaling.

Why not just read report_*_test.json (scripts/metrics_table.py)? Some reports were written with
`evaluate --calibrate-temperature`, so their AUC / ECE / referable metrics use logits / T while the
other models' reports use raw logits -> not comparable. QWK, accuracy and F1 (argmax) are unaffected.
This script calls the SAME grading_metrics() the evaluator uses, on the held-out site's rows only.

Usage:  python scripts/metrics_from_preds.py [--out runs/metrics_from_preds.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils.metrics import grading_metrics  # noqa: E402

FOLDS = ["eyepacs", "ddr", "aptos"]
ENV2ID = {"eyepacs": 0, "ddr": 1, "aptos": 2}
MODELS = {  # name -> (directory, predictions file pattern)
    "Full (CalibCause)": ("runs/stage2", "preds_loeo_heldout_{e}_test.npz"),
    "No-causal": ("runs/ablation_nocausal", "preds_nocausal_loeo_heldout_{e}_test.npz"),
    "Label-free": ("runs/ablation_labelfree", "preds_labelfree_loeo_heldout_{e}_test.npz"),
    "ConvNeXt-T plain": ("runs/baseline", "preds_baseline2_loeo_heldout_{e}_test.npz"),
}
METRICS = ["qwk", "accuracy", "balanced_accuracy", "macro_f1", "auc_ovr", "referable_auc",
           "referable_sensitivity", "referable_specificity", "ece"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for name, (d, pat) in MODELS.items():
        for e in FOLDS:
            path = os.path.join(d, pat.format(e=e))
            row = {"model": name, "heldout": e}
            if os.path.isfile(path):
                z = np.load(path)
                m = z["envs"] == ENV2ID[e]
                met = grading_metrics(z["logits"][m].astype(np.float32), z["labels"][m].astype(np.int64))
                for k in METRICS:
                    row[k] = met.get(k)
            else:
                for k in METRICS:
                    row[k] = None
            rows.append(row)

    def fmt(v):
        return "   --  " if v is None or v != v else f"{v:7.4f}"

    for k in METRICS:
        print(f"\n=== {k} (held-out site, raw logits, T=1) ===")
        print(f"{'model':<20}" + "".join(f"{e:>9}" for e in FOLDS) + f"{'mean':>9}")
        for name in MODELS:
            vals = [next(r[k] for r in rows if r["model"] == name and r["heldout"] == e) for e in FOLDS]
            ok = [v for v in vals if v is not None and v == v]
            mean = sum(ok) / len(ok) if len(ok) == len(FOLDS) else None
            print(f"{name:<20}" + "".join(f"{fmt(v):>9}" for v in vals) + f"{fmt(mean):>9}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "heldout"] + METRICS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
