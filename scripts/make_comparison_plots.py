"""Build individual training curves + a cross-run comparison (QWK overlay,
test accuracy/QWK bars) for any set of Stage-2 runs, all saved into one
output folder.

Usage:
    python scripts/make_comparison_plots.py \
        --runs eyepacs_only:runs/stage2 ddr_only:runs/stage2 \
               aptos_only:runs/stage2 teacher:runs/stage2 \
        --labels "EyePACS-only" "DDR-only" "APTOS-only" "Combined (Teacher)" \
        --out-dir runs/comparison_batch1

Each --runs entry is "<tag>:<dir_containing_history_and_report_json>".
Expects <dir>/history_<tag>.json and <dir>/report_<tag>_test.json to exist.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#937860"]


def load_history(dir_, tag):
    with open(os.path.join(dir_, f"history_{tag}.json")) as f:
        return json.load(f)["history"]


def load_report(dir_, tag):
    with open(os.path.join(dir_, f"report_{tag}_test.json")) as f:
        d = json.load(f)
    return d.get("overall", d)


def save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def plot_single(tag, label, hist, out_dir, color):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ep, [h["train"]["loss"] for h in hist], marker="o", color=color, label="train loss")
    ax0b = ax[0].twinx()
    ax0b.plot(ep, [h["val"]["qwk"] for h in hist], marker="s", color="#C44E52", label="val QWK")
    ax0b.set_ylabel("val QWK", color="#C44E52")
    ax0b.tick_params(axis="y", labelcolor="#C44E52")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss", color=color)
    ax[0].tick_params(axis="y", labelcolor=color)
    ax[0].set_title("Loss vs QWK")

    for key, lab in (("accuracy", "val accuracy"), ("qwk", "val QWK"), ("ece", "val ECE")):
        vals = [h["val"].get(key) for h in hist]
        if all(v is not None for v in vals):
            ax[1].plot(ep, vals, marker=".", label=lab)
    ax[1].set_xlabel("epoch"); ax[1].set_title("Validation metrics")
    ax[1].legend(fontsize=8)

    note = "" if len(hist) >= 25 else f"  (partial: {len(hist)} epochs, interrupted run)"
    fig.suptitle(f"Stage 2 training curve — {label}{note}")
    save(fig, os.path.join(out_dir, f"curves_{tag}.png"))


def plot_overlay(histories, labels, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (tag, hist), label, color in zip(histories.items(), labels, COLORS):
        ep = [h["epoch"] for h in hist]
        qwk = [h["val"]["qwk"] for h in hist]
        ax.plot(ep, qwk, marker="o", label=label, color=color)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation QWK")
    ax.set_title("Validation QWK across runs")
    ax.legend()
    save(fig, os.path.join(out_dir, "comparison_qwk_overlay.png"))


def plot_test_bars(reports, labels, out_dir):
    acc = [r["accuracy"] for r in reports.values()]
    qwk = [r["qwk"] for r in reports.values()]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w / 2 for i in x], acc, width=w, label="Test Accuracy", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], qwk, width=w, label="Test QWK", color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_title("Test-set Accuracy and QWK across runs")
    for i, v in enumerate(acc):
        ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(qwk):
        ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend()
    save(fig, os.path.join(out_dir, "test_comparison_bars.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                     help="tag:dir pairs, e.g. eyepacs_only:runs/stage2")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    assert len(args.runs) == len(args.labels), "one label per run"
    os.makedirs(args.out_dir, exist_ok=True)

    histories, reports = {}, {}
    for run in args.runs:
        tag, dir_ = run.split(":")
        histories[tag] = load_history(dir_, tag)
        reports[tag] = load_report(dir_, tag)

    for (tag, hist), label, color in zip(histories.items(), args.labels, COLORS):
        plot_single(tag, label, hist, args.out_dir, color)

    plot_overlay(histories, args.labels, args.out_dir)
    plot_test_bars(reports, args.labels, args.out_dir)


if __name__ == "__main__":
    main()
