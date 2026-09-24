"""Build the base-paper (QGDR) vs this-work comparison table + bar charts.

QGDR [1] reports (Frontiers in Medicine, 2026):
    EyeQ (~EyePACS quality-labelled subset): 78.32% accuracy, QWK 0.6863
    DDR:                                     80.85% accuracy, QWK 0.8231

Our EyePACS split is the *full* labelled set (not just the EyeQ-annotated
subset QGDR trains on), so the comparison is reported per-environment and
flagged accordingly -- an exact re-run of QGDR's own split is out of scope,
but the same backbone, metric (QWK) and protocol are used.

Reads whatever report_*_test.json files exist in the run directories you
point it at and assembles:
  1. a markdown table (paste straight into the paper)
  2. a grouped bar chart, accuracy and QWK side by side against QGDR

Usage
-----
python scripts/compare_to_qgdr.py \
    --teacher runs/stage2/report_teacher_test.json \
    --labelfree runs/ablation_labelfree/report_labelfree_test.json \
    --nocausal runs/ablation_nocausal/report_nocausal_test.json \
    --student runs/student/report_student_test.json \
    --out runs/comparison
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.utils.plotting import PALETTE, new_fig, save

QGDR = {
    "EyeQ (QGDR)": {"accuracy": 0.7832, "qwk": 0.6863},
    "DDR (QGDR)": {"accuracy": 0.8085, "qwk": 0.8231},
}


def load(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def rows_from_report(name: str, report: dict, per_env: bool = True) -> list[dict]:
    rows = [{"run": name, "split": "overall", **{k: v for k, v in
             report["overall"].items() if k != "confusion_matrix"}}]
    if per_env and "per_environment" in report:
        for env, m in report["per_environment"].items():
            rows.append({"run": name, "split": env,
                        **{k: v for k, v in m.items() if k != "confusion_matrix"}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None, help="report_teacher_test.json")
    ap.add_argument("--labelfree", default=None, help="report_labelfree_test.json")
    ap.add_argument("--nocausal", default=None, help="report_nocausal_test.json")
    ap.add_argument("--student", default=None, help="report_student_test.json")
    ap.add_argument("--loeo", action="append", default=[],
                    help="path[:label] to a LOEO report, repeatable")
    ap.add_argument("--out", default="runs/comparison")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    named = {"teacher (ours, full model)": args.teacher,
            "label-free gating (Module 3 ablation)": args.labelfree,
            "no causal block (Module 1 ablation)": args.nocausal,
            "distilled student (Module 2)": args.student}
    for name, path in named.items():
        rep = load(path)
        if rep is None:
            print(f"  [skip] {name}: {path} not found")
            continue
        rows += rows_from_report(name, rep)

    for entry in args.loeo:
        path, _, label = entry.partition(":")
        label = label or os.path.basename(path)
        rep = load(path)
        if rep is None:
            print(f"  [skip] LOEO {label}: {path} not found")
            continue
        rows += rows_from_report(f"LOEO -- {label}", rep, per_env=True)

    for name, m in QGDR.items():
        rows.append({"run": "QGDR (base paper)", "split": name, **m})

    df = pd.DataFrame(rows)
    cols = ["run", "split", "accuracy", "qwk", "macro_f1", "auc_ovr",
           "referable_auc", "ece"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    csv_path = os.path.join(args.out, "comparison_table.csv")
    md_path = os.path.join(args.out, "comparison_table.md")
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df.round(4).to_markdown(index=False))
    print(df.round(4).to_string(index=False))
    print(f"\nwrote {csv_path}\nwrote {md_path}")

    # bar chart: our overall numbers vs the two QGDR anchors
    plot_df = df[df["split"].isin(["overall", "eyepacs", "ddr", "aptos",
                                   "EyeQ (QGDR)", "DDR (QGDR)"])].copy()
    if len(plot_df):
        fig, ax = new_fig(1, 2, figsize=(12, 4.5))
        x = np.arange(len(plot_df))
        ax[0].bar(x, plot_df["accuracy"], color=[PALETTE.get(s, "#888888")
                                                 for s in plot_df["split"]])
        ax[0].set_xticks(x); ax[0].set_xticklabels(
            plot_df["run"] + "\n(" + plot_df["split"] + ")", rotation=60, ha="right",
            fontsize=7)
        ax[0].set_ylabel("accuracy"); ax[0].set_title("accuracy")

        ax[1].bar(x, plot_df["qwk"], color=[PALETTE.get(s, "#888888")
                                            for s in plot_df["split"]])
        ax[1].set_xticks(x); ax[1].set_xticklabels(
            plot_df["run"] + "\n(" + plot_df["split"] + ")", rotation=60, ha="right",
            fontsize=7)
        ax[1].set_ylabel("QWK"); ax[1].set_title("quadratic weighted kappa")
        save(fig, os.path.join(args.out, "comparison_bars.png"))


if __name__ == "__main__":
    main()
