"""Held-out (unseen-site) metrics for every LOEO model in one table -- reads the existing
report_*_test.json files only (no GPU, no predictions needed).

Each report holds the metrics of ALL test environments; the unseen-site number is
report["per_environment"][<held-out env>] (the DDR key is "ddr", or "idrid" in older reports).

Usage:  python scripts/metrics_table.py [--out runs/metrics_table.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

FOLDS = ["eyepacs", "ddr", "aptos"]
MODELS = {  # name -> (directory, report file pattern)
    "Full (CalibCause)": ("runs/stage2", "report_loeo_heldout_{e}_test.json"),
    "No-causal": ("runs/ablation_nocausal", "report_nocausal_loeo_heldout_{e}_test.json"),
    "Label-free": ("runs/ablation_labelfree", "report_labelfree_loeo_heldout_{e}_test.json"),
    "ConvNeXt-T plain": ("runs/baseline", "report_baseline2_loeo_heldout_{e}_test.json"),
}
METRICS = ["qwk", "accuracy", "balanced_accuracy", "macro_f1", "auc_ovr", "referable_auc", "referable_sensitivity", "referable_specificity", "ece"]


def held_out(report: dict, env: str):
    per = report.get("per_environment", {})
    for key in ([env, "idrid"] if env == "ddr" else [env]):
        if key in per:
            return per[key]
    return None


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
                with open(path) as f:
                    m = held_out(json.load(f), e)
                for k in METRICS:
                    row[k] = None if m is None else m.get(k)
            else:
                for k in METRICS:
                    row[k] = None
            rows.append(row)

    def fmt(v):
        return "   --  " if v is None or v != v else f"{v:7.4f}"

    for k in METRICS:
        print(f"\n=== {k} (held-out site, test split) ===")
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
