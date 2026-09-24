"""Loss / accuracy / QWK curves from the history_*.json files every training
script writes.

Handles all three history shapes in this repo:
  history_<tag>.json          from train_stage2.py   -> per-epoch train+val
  history_<tag>.json          from train_student.py  -> per-epoch train+val (KD)
  stage1 does not write a history_*.json (it checkpoints every epoch instead);
  use --stage1-log for that curve, parsed straight from the text log.

Examples
--------
# one run
python scripts/plot_curves.py --history runs/stage2/history_teacher.json \
    --out runs/stage2/curves_teacher.png

# overlay several runs on the same axes (e.g. teacher vs label-free ablation)
python scripts/plot_curves.py \
    --history runs/stage2/history_teacher.json:teacher \
    --history runs/ablation_labelfree/history_labelfree.json:labelfree \
    --history runs/ablation_nocausal/history_nocausal.json:nocausal \
    --metric qwk --out runs/comparison_qwk.png

# Stage 1 SSL loss (parsed from the log file, no JSON history is written)
python scripts/plot_curves.py --stage1-log runs/stage1/stage1.log \
    --out runs/stage1/curves_ssl.png
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import re

import pandas as pd

from src.utils.plotting import PALETTE, new_fig, save


def load_history(path: str) -> pd.DataFrame:
    with open(path) as f:
        h = json.load(f)
    rows = h.get("history", h)  # some files are {history: [...]}, tolerate both
    df = pd.json_normalize(rows, sep=".")
    return df


def parse_stage1_log(path: str) -> pd.DataFrame:
    """Stage 1 logs one line per epoch:
        [HH:MM:SS] INFO | epoch 3: reg=0.1234 rank=0.0456
    """
    pat = re.compile(r"epoch (\d+): reg=([\d.]+) rank=([\d.]+)")
    rows = []
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append({"epoch": int(m.group(1)), "reg": float(m.group(2)),
                            "rank": float(m.group(3))})
    return pd.DataFrame(rows)


def plot_single(df: pd.DataFrame, out: str, title: str = ""):
    """Two panels: loss (train vs val) and accuracy+QWK (val)."""
    fig, ax = new_fig(1, 2, figsize=(11, 4.2))
    if "train.loss" in df.columns:
        ax[0].plot(df["epoch"], df["train.loss"], marker="o", label="train",
                  color=PALETTE["train"])
    if "val.accuracy" in df.columns and "train.loss" in df.columns:
        # val split reports metrics, not loss -- show accuracy on a twin axis instead
        pass
    if "val.qwk" in df.columns:
        ax2 = ax[0].twinx()
        ax2.plot(df["epoch"], df["val.qwk"], marker="s", color=PALETTE["val"],
                 label="val QWK")
        ax2.set_ylabel("val QWK", color=PALETTE["val"])
        ax2.tick_params(axis="y", labelcolor=PALETTE["val"])
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss", color=PALETTE["train"])
    ax[0].tick_params(axis="y", labelcolor=PALETTE["train"])
    ax[0].set_title("loss vs QWK")

    for col, lab in (("val.accuracy", "val accuracy"), ("val.qwk", "val QWK"),
                     ("val.macro_f1", "val macro-F1"), ("val.ece", "val ECE")):
        if col in df.columns:
            ax[1].plot(df["epoch"], df[col], marker=".", label=lab)
    ax[1].set_xlabel("epoch"); ax[1].set_title("validation metrics")
    ax[1].legend(fontsize=8)
    fig.suptitle(title)
    save(fig, out)


def plot_overlay(histories: dict[str, pd.DataFrame], metric: str, out: str):
    fig, ax = new_fig(1, 1, figsize=(6.5, 4.5))
    col = f"val.{metric}"
    for name, df in histories.items():
        if col not in df.columns:
            print(f"  [warn] {name} has no column {col}, skipping")
            continue
        ax.plot(df["epoch"], df[col], marker="o", label=name,
               color=PALETTE.get(name))
    ax.set_xlabel("epoch"); ax.set_ylabel(f"val {metric}")
    ax.set_title(f"val {metric} across runs")
    ax.legend()
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="append", default=[],
                    help="path[:label] to a history_*.json; repeat for overlay plots")
    ap.add_argument("--stage1-log", default=None)
    ap.add_argument("--metric", default="qwk",
                    choices=["qwk", "accuracy", "macro_f1", "ece"],
                    help="metric to overlay when >1 --history is given")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.stage1_log:
        df = parse_stage1_log(args.stage1_log)
        fig, ax = new_fig(1, 1, figsize=(6, 4.2))
        ax.plot(df["epoch"], df["reg"], marker="o", label="severity regression (BCE)")
        ax.plot(df["epoch"], df["rank"], marker="s", label="pairwise ranking")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title("Stage 1 -- self-supervised quality pretraining")
        ax.legend()
        save(fig, args.out)
        return

    if not args.history:
        raise SystemExit("give --history <path[:label]> or --stage1-log <path>")

    parsed = {}
    for h in args.history:
        path, _, label = h.partition(":")
        label = label or path.split("/")[-1].replace("history_", "").replace(".json", "")
        parsed[label] = load_history(path)

    if len(parsed) == 1:
        (label, df), = parsed.items()
        plot_single(df, args.out, title=label)
    else:
        plot_overlay(parsed, args.metric, args.out)


if __name__ == "__main__":
    main()
