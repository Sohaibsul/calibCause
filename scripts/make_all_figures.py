"""
make_all_figures.py  --  generate EVERY paper figure from the run outputs.

Produces, into --out-dir (default runs/figures/):
  1. curves_<tag>.png            per-run loss + val-QWK curve (from history JSON
                                 or, if missing, rebuilt from a *_console.log)
  2. comparison_qwk_overlay.png  val-QWK vs epoch, all four main runs overlaid
  3. test_bars_persource.png     test Acc + QWK bars: ddr/aptos/eyepacs/teacher
  4. loeo_summary.png            LOEO held-out QWK vs in-domain ceiling
  5. ablation_bars.png           teacher vs no-causal vs label-free (QWK)
  6. student_vs_teacher.png      distillation trade-off (QWK, params if present)
  7. confusion_<tag>.png         confusion-matrix heatmap for each report
  8. reliability.png             reliability diagram (from uncertainty/ dir)
  9. deferral.png                accuracy-vs-coverage (from uncertainty/ dir)
 10. routing_share.png           dominant-expert share (from routing/ dir)

It is defensive: any figure whose inputs are missing is skipped with a
printed note, so a partial run still yields whatever it can.

Usage:
    python scripts/make_all_figures.py \
        --runs-dir runs --out-dir runs/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#937860"]
LABELS = {
    "ddr_only": "DDR-only", "aptos_only": "APTOS-only",
    "eyepacs_only": "EyePACS-only", "teacher": "Combined",
    "labelfree": "Label-free", "nocausal": "No-causal", "student": "Student",
    "loeo_heldout_eyepacs": "Held-out EyePACS",
    "loeo_heldout_ddr": "Held-out DDR",
    "loeo_heldout_aptos": "Held-out APTOS",
}
GRADES = ["0", "1", "2", "3", "4"]


def savefig(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ------------------------------------------------------------------ helpers
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def rebuild_history_from_log(log_path):
    """Parse a *_console.log for stage2/student epoch lines."""
    if not os.path.isfile(log_path):
        return None
    txt = open(log_path).read().splitlines()
    ep = re.compile(r"epoch (\d+): .*?loss=([\d.]+).*?acc=([\d.]+)")
    valp = re.compile(r"val acc=([\d.]+) qwk=([\d.]+)(?: ece=([\d.]+))?")
    # student format: "epoch N: loss=.. teacher-agreement=.. val acc=.. qwk=.."
    hist = []
    for i, line in enumerate(txt):
        m = ep.search(line)
        if not m:
            m2 = re.search(r"epoch (\d+): loss=([\d.]+) teacher-agreement=[\d.]+ "
                           r"val acc=([\d.]+) qwk=([\d.]+)", line)
            if m2:
                e, loss, acc, qwk = m2.groups()
                hist.append({"epoch": int(e), "train": {"loss": float(loss)},
                             "val": {"accuracy": float(acc), "qwk": float(qwk)}})
            continue
        # look at next line for val metrics
        vm = valp.search(txt[i + 1]) if i + 1 < len(txt) else None
        if vm:
            e, loss, tacc = m.groups()
            vacc, vqwk, vece = vm.groups()
            hist.append({"epoch": int(e),
                         "train": {"loss": float(loss), "acc": float(tacc)},
                         "val": {"accuracy": float(vacc), "qwk": float(vqwk),
                                 "ece": float(vece) if vece else None}})
    return hist or None


def get_history(runs_dir, tag):
    for cand in (f"{runs_dir}/stage2/history_{tag}.json",
                 f"{runs_dir}/ablation_{tag}/history_{tag}.json",
                 f"{runs_dir}/student/history_{tag}.json"):
        d = load_json(cand)
        if d and d.get("history"):
            return d["history"]
    # fall back to console logs
    for log in glob.glob(f"{runs_dir}/**/*{tag}*console*.log", recursive=True) + \
               glob.glob(f"{runs_dir}/*console*.log"):
        h = rebuild_history_from_log(log)
        if h:
            return h
    return None


def get_report(runs_dir, tag):
    for cand in (f"{runs_dir}/stage2/report_{tag}_test.json",
                 f"{runs_dir}/ablation_{tag}/report_{tag}_test.json",
                 f"{runs_dir}/student/report_{tag}_test.json"):
        d = load_json(cand)
        if d:
            return d.get("overall", d), d
    return None, None


# ------------------------------------------------------------------ figures
def fig_curves(runs_dir, out):
    for tag in LABELS:
        hist = get_history(runs_dir, tag)
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(figsize=(6, 4))
        if "loss" in hist[0].get("train", {}):
            ax.plot(ep, [h["train"]["loss"] for h in hist], marker="o",
                    color=PALETTE[0], label="train loss")
            ax.set_ylabel("train loss", color=PALETTE[0])
            ax.tick_params(axis="y", labelcolor=PALETTE[0])
        axb = ax.twinx()
        axb.plot(ep, [h["val"]["qwk"] for h in hist], marker="s",
                 color=PALETTE[4], label="val QWK")
        axb.set_ylabel("val QWK", color=PALETTE[4])
        axb.tick_params(axis="y", labelcolor=PALETTE[4])
        ax.set_xlabel("epoch")
        note = "" if len(hist) >= 25 else f"  (partial: {len(hist)} ep)"
        ax.set_title(f"{LABELS[tag]}{note}")
        savefig(fig, f"{out}/curves_{tag}.png")


def fig_overlay(runs_dir, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    n = 0
    for tag in ("ddr_only", "aptos_only", "eyepacs_only", "teacher"):
        hist = get_history(runs_dir, tag)
        if not hist:
            continue
        ax.plot([h["epoch"] for h in hist], [h["val"]["qwk"] for h in hist],
                marker="o", color=PALETTE[n], label=LABELS[tag])
        n += 1
    if n:
        ax.set_xlabel("epoch"); ax.set_ylabel("validation QWK")
        ax.set_title("Validation QWK across runs"); ax.legend()
        savefig(fig, f"{out}/comparison_qwk_overlay.png")
    else:
        plt.close(fig)


def fig_test_bars(runs_dir, out):
    tags = ["ddr_only", "aptos_only", "eyepacs_only", "teacher"]
    acc, qwk, names = [], [], []
    for t in tags:
        o, _ = get_report(runs_dir, t)
        if o:
            acc.append(o["accuracy"]); qwk.append(o["qwk"]); names.append(LABELS[t])
    if not names:
        return
    x = range(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=PALETTE[0])
    ax.bar([i + w/2 for i in x], qwk, w, label="QWK", color=PALETTE[1])
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(qwk):
        ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_title("Test accuracy and QWK"); ax.legend()
    savefig(fig, f"{out}/test_bars_persource.png")


def fig_loeo(runs_dir, out):
    tags = ["loeo_heldout_eyepacs", "loeo_heldout_ddr", "loeo_heldout_aptos"]
    qwk, names = [], []
    for t in tags:
        o, _ = get_report(runs_dir, t)
        if o:
            qwk.append(o["qwk"]); names.append(LABELS[t])
    if not names:
        return
    ceiling = None
    o, _ = get_report(runs_dir, "teacher")
    if o:
        ceiling = o["qwk"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, qwk, color=PALETTE[2])
    for b, v in zip(bars, qwk):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    if ceiling:
        ax.axhline(ceiling, ls="--", color=PALETTE[4],
                   label=f"in-domain ceiling ({ceiling:.3f})")
        ax.legend()
    ax.set_ylim(0, 1.05); ax.set_ylabel("QWK on held-out source")
    ax.set_title("Cross-environment generalisation (LOEO)")
    savefig(fig, f"{out}/loeo_summary.png")


def fig_ablation(runs_dir, out):
    order = [("teacher", "Full teacher"), ("nocausal", "No-causal (Mod.1)"),
             ("labelfree", "Label-free (Mod.3)")]
    qwk, names = [], []
    for t, lab in order:
        o, _ = get_report(runs_dir, t)
        if o:
            qwk.append(o["qwk"]); names.append(lab)
    if len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, qwk, color=[PALETTE[0], PALETTE[1], PALETTE[2]][:len(names)])
    for b, v in zip(bars, qwk):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("test QWK")
    ax.set_title("Ablation: contribution of each module")
    savefig(fig, f"{out}/ablation_bars.png")


def fig_student(runs_dir, out):
    ot, _ = get_report(runs_dir, "teacher")
    os_, _ = get_report(runs_dir, "student")
    if not (ot and os_):
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    names = ["Teacher", "Student"]
    qwk = [ot["qwk"], os_["qwk"]]
    acc = [ot["accuracy"], os_["accuracy"]]
    x = range(2); w = 0.35
    ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=PALETTE[0])
    ax.bar([i + w/2 for i in x], qwk, w, label="QWK", color=PALETTE[1])
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(qwk):
        ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_title("Distillation trade-off"); ax.legend()
    savefig(fig, f"{out}/student_vs_teacher.png")


def fig_confusions(runs_dir, out):
    for tag in LABELS:
        _, full = get_report(runs_dir, tag)
        if not full:
            continue
        cm = full.get("confusion_matrix") or full.get("confusion")
        if cm is None:
            continue
        cm = np.array(cm, dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(5, 4.3))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(GRADES))); ax.set_xticklabels(GRADES)
        ax.set_yticks(range(len(GRADES))); ax.set_yticklabels(GRADES)
        ax.set_xlabel("predicted grade"); ax.set_ylabel("true grade")
        ax.set_title(f"Confusion matrix — {LABELS[tag]}")
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        savefig(fig, f"{out}/confusion_{tag}.png")


def fig_reliability_deferral(runs_dir, out):
    udir = f"{runs_dir}/stage2/uncertainty"
    rel = load_json(f"{udir}/reliability.json") or load_json(f"{udir}/summary.json")
    # reliability
    if rel and "bins" in rel:
        conf = [b["confidence"] for b in rel["bins"]]
        acc = [b["accuracy"] for b in rel["bins"]]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
        ax.plot(conf, acc, marker="o", color=PALETTE[0], label="model")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title("Reliability diagram"); ax.legend()
        savefig(fig, f"{out}/reliability.png")
    # deferral
    def_ = load_json(f"{udir}/deferral.json")
    if def_ and "coverage" in def_:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(def_["coverage"], def_["accuracy"], marker=".", color=PALETTE[2])
        ax.set_xlabel("coverage"); ax.set_ylabel("accuracy on retained set")
        ax.set_title("Accuracy vs coverage (deferral)")
        savefig(fig, f"{out}/deferral.png")
    if not rel and not def_:
        print("  [skip] reliability/deferral JSON not found in", udir)


def fig_routing(runs_dir, out):
    rdir = f"{runs_dir}/stage2/routing"
    share = load_json(f"{rdir}/share.json") or load_json(f"{rdir}/summary.json")
    if not share:
        print("  [skip] routing share JSON not found in", rdir)
        return
    d = share.get("dominant_share", share)
    if not isinstance(d, dict):
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    names = list(d.keys()); vals = list(d.values())
    bars = ax.bar(names, vals, color=PALETTE[:len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("dominant-expert share")
    ax.set_title("Expert routing (dominant share)")
    plt.xticks(rotation=15)
    savefig(fig, f"{out}/routing_share.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="runs/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("1. per-run curves");            fig_curves(args.runs_dir, args.out_dir)
    print("2. QWK overlay");               fig_overlay(args.runs_dir, args.out_dir)
    print("3. per-source test bars");      fig_test_bars(args.runs_dir, args.out_dir)
    print("4. LOEO summary");              fig_loeo(args.runs_dir, args.out_dir)
    print("5. ablation bars");             fig_ablation(args.runs_dir, args.out_dir)
    print("6. student vs teacher");        fig_student(args.runs_dir, args.out_dir)
    print("7. confusion matrices");        fig_confusions(args.runs_dir, args.out_dir)
    print("8-9. reliability + deferral");  fig_reliability_deferral(args.runs_dir, args.out_dir)
    print("10. routing share");            fig_routing(args.runs_dir, args.out_dir)
    print(f"\nAll available figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
