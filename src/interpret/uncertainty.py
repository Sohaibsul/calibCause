"""Contribution 4 (the 'Calib' in CalibCause) -- calibrated uncertainty analysis.

Produces, from a prediction dump written by src.engine.evaluate:
  * reliability diagram + ECE, before and after temperature scaling
  * risk-coverage curve: accuracy as a function of how many low-confidence
    cases the model is allowed to refer to a human reader
  * confidence vs image quality, and confidence vs environment -- i.e. whether
    the uncertainty signal actually tracks the two shifts the rest of the
    pipeline is built around
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from ..data.datasets import ID2ENV
from ..utils import TemperatureScaler, expected_calibration_error


def _softmax(x: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(x).float(), dim=-1).numpy()


def risk_coverage(probs: np.ndarray, labels: np.ndarray, steps: int = 21) -> pd.DataFrame:
    """Sort by confidence, keep the most confident fraction, measure accuracy.

    The deferral point a screening programme actually cares about is 'how much
    accuracy do I get if a clinician reads the least confident 20%?', which is
    what this curve reports.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    order = np.argsort(-conf)
    correct, conf = correct[order], conf[order]
    rows = []
    for cov in np.linspace(0.1, 1.0, steps):
        k = max(1, int(round(cov * len(correct))))
        rows.append({"coverage": float(k / len(correct)),
                     "accuracy": float(correct[:k].mean()),
                     "risk": float(1 - correct[:k].mean()),
                     "confidence_threshold": float(conf[k - 1])})
    return pd.DataFrame(rows)


def analyse(npz_path: str, out_dir: str, val_npz: str | None = None,
            make_plots: bool = True) -> dict:
    d = np.load(npz_path)
    logits, labels = d["logits"], d["labels"]

    # temperature fitted on a *held-out* split when available; falling back to
    # the test split itself is reported so the number is not read as clean
    fit_src = "validation"
    if val_npz and os.path.isfile(val_npz):
        v = np.load(val_npz)
        ts = TemperatureScaler().fit(v["logits"], v["labels"])
    else:
        fit_src = "test (no val dump given -- optimistic)"
        ts = TemperatureScaler().fit(logits, labels)
    T = ts.T

    p_raw, p_cal = _softmax(logits), _softmax(logits / T)
    ece_raw, bins_raw = expected_calibration_error(p_raw, labels)
    ece_cal, bins_cal = expected_calibration_error(p_cal, labels)

    os.makedirs(out_dir, exist_ok=True)
    rc = risk_coverage(p_cal, labels)
    rc.to_csv(os.path.join(out_dir, "risk_coverage.csv"), index=False)
    pd.DataFrame(bins_cal).to_csv(os.path.join(out_dir, "reliability_bins.csv"), index=False)

    summary = {
        "temperature": T, "temperature_fitted_on": fit_src,
        "ece_uncalibrated": ece_raw, "ece_calibrated": ece_cal,
        "accuracy": float((p_cal.argmax(1) == labels).mean()),
        "mean_confidence": float(p_cal.max(1).mean()),
        "accuracy_at_80pct_coverage": float(
            rc.iloc[(rc.coverage - 0.8).abs().argmin()]["accuracy"]),
        "accuracy_at_90pct_coverage": float(
            rc.iloc[(rc.coverage - 0.9).abs().argmin()]["accuracy"]),
    }
    if "quality_score" in d:
        q = d["quality_score"]
        summary["corr_confidence_vs_quality"] = float(np.corrcoef(p_cal.max(1), q)[0, 1])
    if "envs" in d:
        summary["mean_confidence_by_env"] = {
            ID2ENV[int(e)]: float(p_cal.max(1)[d["envs"] == e].mean())
            for e in np.unique(d["envs"])}
        summary["ece_by_env"] = {
            ID2ENV[int(e)]: expected_calibration_error(p_cal[d["envs"] == e],
                                                       labels[d["envs"] == e])[0]
            for e in np.unique(d["envs"]) if (d["envs"] == e).sum() >= 20}

    pd.Series(summary).to_json(os.path.join(out_dir, "uncertainty_summary.json"), indent=2)

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for bins, lab in ((bins_raw, f"raw (ECE={ece_raw:.3f})"),
                          (bins_cal, f"T={T:.2f} (ECE={ece_cal:.3f})")):
            b = pd.DataFrame(bins).dropna()
            ax[0].plot(b["conf"], b["acc"], marker="o", label=lab)
        ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy")
        ax[0].set_title("reliability diagram"); ax[0].legend()
        ax[1].plot(rc["coverage"], rc["accuracy"], marker=".")
        ax[1].set_xlabel("coverage (fraction auto-graded)")
        ax[1].set_ylabel("accuracy on covered cases")
        ax[1].set_title("risk-coverage (deferral) curve")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=160)
        plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds_*_test.npz")
    ap.add_argument("--val-preds", default=None, help="preds_*_val.npz to fit T on")
    ap.add_argument("--out", default="runs/uncertainty")
    args = ap.parse_args()
    s = analyse(args.preds, args.out, args.val_preds)
    print(f"T={s['temperature']:.3f}  ECE {s['ece_uncalibrated']:.4f} -> {s['ece_calibrated']:.4f}")
    print(f"acc@80% coverage = {s['accuracy_at_80pct_coverage']:.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
