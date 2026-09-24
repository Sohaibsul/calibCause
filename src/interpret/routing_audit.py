"""Module 2 -- clinician-facing routing audit.

Consumes the prediction dump written by src.engine.evaluate and answers:
  * which expert dominated each prediction?
  * does the gate actually respond to image quality (as designed) or has it
    collapsed onto one expert?
  * does expert dominance track DR grade, environment, or only quality?
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..data.datasets import ID2ENV

EXPERT_NAMES = ("fine_lesion", "vascular_lesion", "severe_dr")


def audit(npz_path: str, out_dir: str, make_plots: bool = True) -> dict:
    d = np.load(npz_path)
    if "gate_weights" not in d:
        raise ValueError("prediction dump has no gate weights (student model?)")
    g, q = d["gate_weights"], d["quality_score"]
    y, env = d["labels"], d["envs"]
    pred = d["logits"].argmax(1)
    dom = g.argmax(1)

    df = pd.DataFrame({"grade": y, "pred": pred, "env": [ID2ENV[int(e)] for e in env],
                       "quality": q, "dominant_expert": [EXPERT_NAMES[i] for i in dom]})
    for i, n in enumerate(EXPERT_NAMES):
        df[f"gate_{n}"] = g[:, i]

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "routing_per_sample.csv"), index=False)

    gate_cols = [f"gate_{n}" for n in EXPERT_NAMES]
    summary = {
        "dominant_expert_share": df["dominant_expert"].value_counts(normalize=True).to_dict(),
        "mean_gate": df[gate_cols].mean().to_dict(),
        "gate_entropy_mean": float(-(g * np.log(np.clip(g, 1e-8, 1))).sum(1).mean()),
        "corr_gate_vs_quality": {n: float(np.corrcoef(df[f"gate_{n}"], q)[0, 1])
                                 for n in EXPERT_NAMES},
        "mean_gate_by_grade": df.groupby("grade")[gate_cols].mean().to_dict(),
        "mean_gate_by_env": df.groupby("env")[gate_cols].mean().to_dict(),
        "accuracy_by_dominant_expert":
            df.assign(correct=(df.grade == df.pred))
              .groupby("dominant_expert")["correct"].agg(["mean", "count"]).to_dict(),
    }
    # quality tertiles -- the interesting cut, given quality-severity coupling
    df["quality_stratum"] = pd.qcut(df["quality"], 3,
                                    labels=["low", "mid", "high"], duplicates="drop")
    summary["mean_gate_by_quality_stratum"] = \
        df.groupby("quality_stratum", observed=True)[gate_cols].mean().to_dict()
    summary["referable_prevalence_by_quality_stratum"] = \
        df.assign(ref=(df.grade >= 2)).groupby("quality_stratum",
                                               observed=True)["ref"].mean().to_dict()

    pd.Series(summary).to_json(os.path.join(out_dir, "routing_summary.json"), indent=2)

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        df.groupby("grade")[gate_cols].mean().plot.bar(ax=axes[0], rot=0)
        axes[0].set_title("mean gate weight by DR grade")
        df.groupby("quality_stratum", observed=True)[gate_cols].mean().plot.bar(ax=axes[1], rot=0)
        axes[1].set_title("mean gate weight by quality stratum")
        for i, n in enumerate(EXPERT_NAMES):
            axes[2].scatter(q[::max(1, len(q) // 2000)], g[::max(1, len(q) // 2000), i],
                            s=3, alpha=0.3, label=n)
        axes[2].set_xlabel("quality score")
        axes[2].set_ylabel("gate weight")
        axes[2].legend()
        axes[2].set_title("gate response to quality")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "routing_audit.png"), dpi=160)
        plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds_*.npz from src.engine.evaluate")
    ap.add_argument("--out", default="runs/routing_audit")
    args = ap.parse_args()
    s = audit(args.preds, args.out)
    print(f"dominant-expert share: {s['dominant_expert_share']}")
    print(f"gate-quality correlation: {s['corr_gate_vs_quality']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
