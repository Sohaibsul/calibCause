"""
make_paper_figures.py  --  build ALL paper figures from run outputs and
organise them into stage-wise folders under --out-dir.

Output layout:
  figures/
    01_per_source/    curves + confusion + acc/QWK bars  (ddr, aptos, eyepacs)
    02_combined/      teacher curve, confusion, per-env bars
    03_loeo/          LOEO summary + per-held-out confusion
    04_ablations/     teacher vs no-causal vs label-free (QWK/ECE bars)
    05_student/       teacher-vs-student trade-off, student curve
    06_calibration/   reliability diagram, deferral curve
    07_routing/       dominant-expert share
    08_external/      DeepDRiD + IDRiD bars + confusion
    09_overall/       master QWK comparison across every run

Defensive: any figure whose inputs are missing is skipped with a note,
so a partial set of results still yields whatever it can.

Usage:
    python scripts/make_paper_figures.py --runs-dir runs --out-dir figures
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

PAL = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52", "#937860", "#DA8BC3"]
GRADES = ["0", "1", "2", "3", "4"]

# tag -> (folder, nice label)
TAGS = {
    "ddr_only":              ("01_per_source", "DDR-only"),
    "aptos_only":            ("01_per_source", "APTOS-only"),
    "eyepacs_only":          ("01_per_source", "EyePACS-only"),
    "teacher":               ("02_combined",   "Combined (teacher)"),
    "loeo_heldout_eyepacs":  ("03_loeo",        "Held-out EyePACS"),
    "loeo_heldout_ddr":      ("03_loeo",        "Held-out DDR"),
    "loeo_heldout_aptos":    ("03_loeo",        "Held-out APTOS"),
    "labelfree":             ("04_ablations",   "Label-free"),
    "nocausal":              ("04_ablations",   "No-causal"),
    "student":               ("05_student",     "Student"),
    "deepdrid_teacher":      ("08_external",    "DeepDRiD"),
    "idrid_teacher":         ("08_external",    "IDRiD"),
}


def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def find_report(runs, tag):
    # test reports OR external reports (report_<tag>_<split>.json)
    for p in glob.glob(f"{runs}/**/report_{tag}_*.json", recursive=True):
        d = load_json(p)
        if d:
            return d.get("overall", d), d
    return None, None


def find_history(runs, tag):
    for p in glob.glob(f"{runs}/**/history_{tag}.json", recursive=True):
        d = load_json(p)
        if d and d.get("history"):
            return d["history"]
    return None


# ---------------------------------------------------------------- curves
def make_curves(runs, out):
    for tag, (folder, label) in TAGS.items():
        hist = find_history(runs, tag)
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        fig, ax = plt.subplots(figsize=(6, 4))
        if "train" in hist[0] and "loss" in hist[0]["train"]:
            ax.plot(ep, [h["train"]["loss"] for h in hist], marker="o",
                    color=PAL[0], label="train loss")
            ax.set_ylabel("train loss", color=PAL[0])
            ax.tick_params(axis="y", labelcolor=PAL[0])
        axb = ax.twinx()
        axb.plot(ep, [h["val"]["qwk"] for h in hist], marker="s",
                 color=PAL[4], label="val QWK")
        axb.set_ylabel("val QWK", color=PAL[4])
        axb.tick_params(axis="y", labelcolor=PAL[4])
        ax.set_xlabel("epoch")
        note = "" if len(hist) >= 25 else f"  ({len(hist)} ep)"
        ax.set_title(f"Training curve — {label}{note}")
        savefig(fig, f"{out}/{folder}/curve_{tag}.png")


# ---------------------------------------------------------------- confusions
def make_confusions(runs, out):
    for tag, (folder, label) in TAGS.items():
        _, full = find_report(runs, tag)
        if not full:
            continue
        cm = full.get("confusion_matrix")
        if cm is None:
            continue
        cm = np.array(cm, dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(GRADES)
        ax.set_yticks(range(5)); ax.set_yticklabels(GRADES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion — {label}")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        savefig(fig, f"{out}/{folder}/confusion_{tag}.png")


# ---------------------------------------------------------------- per-source bars
def make_persource_bars(runs, out):
    tags = ["ddr_only", "aptos_only", "eyepacs_only"]
    acc, qwk, names = [], [], []
    for t in tags:
        o, _ = find_report(runs, t)
        if o:
            acc.append(o["accuracy"]); qwk.append(o["qwk"])
            names.append(TAGS[t][1])
    if not names:
        return
    x = range(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=PAL[0])
    ax.bar([i + w/2 for i in x], qwk, w, label="QWK", color=PAL[1])
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(qwk):
        ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_title("Per-source test performance"); ax.legend()
    savefig(fig, f"{out}/01_per_source/bars_per_source.png")


# ---------------------------------------------------------------- per-env (teacher)
def make_perenv(runs, out):
    _, full = find_report(runs, "teacher")
    if not full or "per_environment" not in full:
        return
    pe = full["per_environment"]
    envs = list(pe.keys())
    qwk = [pe[e].get("qwk", np.nan) for e in envs]
    acc = [pe[e].get("accuracy", np.nan) for e in envs]
    x = range(len(envs)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=PAL[0])
    ax.bar([i + w/2 for i in x], qwk, w, label="QWK", color=PAL[1])
    ax.set_xticks(list(x)); ax.set_xticklabels(envs)
    ax.set_ylim(0, 1.05); ax.set_title("Combined teacher — per environment")
    ax.legend()
    savefig(fig, f"{out}/02_combined/bars_per_env.png")


# ---------------------------------------------------------------- LOEO summary
def make_loeo(runs, out):
    tags = ["loeo_heldout_eyepacs", "loeo_heldout_ddr", "loeo_heldout_aptos"]
    qwk, names = [], []
    for t in tags:
        o, _ = find_report(runs, t)
        if o:
            qwk.append(o["qwk"]); names.append(TAGS[t][1])
    if not names:
        return
    o, _ = find_report(runs, "teacher")
    ceiling = o["qwk"] if o else None
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, qwk, color=PAL[2])
    for b, v in zip(bars, qwk):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    if ceiling:
        ax.axhline(ceiling, ls="--", color=PAL[4],
                   label=f"in-domain ceiling ({ceiling:.3f})")
        ax.legend()
    ax.set_ylim(0, 1.05); ax.set_ylabel("QWK on unseen source")
    ax.set_title("Cross-environment generalisation (LOEO)")
    savefig(fig, f"{out}/03_loeo/loeo_summary.png")


# ---------------------------------------------------------------- ablation bars
def make_ablation(runs, out):
    order = [("teacher", "Full teacher"), ("nocausal", "No-causal"),
             ("labelfree", "Label-free")]
    qwk, ece, names = [], [], []
    for t, lab in order:
        o, _ = find_report(runs, t)
        if o:
            qwk.append(o["qwk"]); ece.append(o["ece"]); names.append(lab)
    if len(names) < 2:
        return
    x = range(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - w/2 for i in x], qwk, w, label="QWK", color=PAL[0])
    ax.bar([i + w/2 for i in x], ece, w, label="ECE", color=PAL[3])
    for i, v in enumerate(qwk):
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(ece):
        ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_title("Ablation (QWK ↑, ECE ↓)"); ax.legend()
    savefig(fig, f"{out}/04_ablations/ablation_bars.png")


# ---------------------------------------------------------------- student
def make_student(runs, out):
    ot, _ = find_report(runs, "teacher")
    os_, _ = find_report(runs, "student")
    if not (ot and os_):
        return
    names = ["Teacher", "Student"]
    acc = [ot["accuracy"], os_["accuracy"]]
    qwk = [ot["qwk"], os_["qwk"]]
    ece = [ot["ece"], os_["ece"]]
    x = range(2); w = 0.27
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar([i - w for i in x], acc, w, label="Accuracy", color=PAL[0])
    ax.bar(list(x),          qwk, w, label="QWK", color=PAL[1])
    ax.bar([i + w for i in x], ece, w, label="ECE", color=PAL[3])
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05); ax.set_title("Distillation trade-off"); ax.legend()
    savefig(fig, f"{out}/05_student/student_vs_teacher.png")


# ---------------------------------------------------------------- calibration
def make_calibration(runs, out):
    udir_candidates = glob.glob(f"{runs}/**/uncertainty", recursive=True)
    for udir in udir_candidates:
        summ = load_json(f"{udir}/uncertainty_summary.json")
        if not summ:
            continue
        # reliability (if bins present)
        bins = summ.get("reliability") or summ.get("bins")
        if bins:
            conf = [b.get("conf", b.get("confidence")) for b in bins if b.get("n", 1)]
            acc = [b.get("acc", b.get("accuracy")) for b in bins if b.get("n", 1)]
            conf = [c for c in conf if c is not None]
            acc = [a for a in acc if a is not None]
            if conf and acc:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
                ax.plot(conf, acc, marker="o", color=PAL[0], label="model")
                ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
                ax.set_title("Reliability diagram"); ax.legend()
                savefig(fig, f"{out}/06_calibration/reliability.png")
        # deferral (accuracy vs coverage)
        defr = summ.get("deferral")
        if defr and "coverage" in defr:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.plot(defr["coverage"], defr["accuracy"], marker=".", color=PAL[2])
            ax.set_xlabel("coverage"); ax.set_ylabel("accuracy on retained set")
            ax.set_title("Accuracy vs coverage (deferral)")
            savefig(fig, f"{out}/06_calibration/deferral.png")


# ---------------------------------------------------------------- routing
def make_routing(runs, out):
    for rdir in glob.glob(f"{runs}/**/routing", recursive=True):
        summ = load_json(f"{rdir}/routing_summary.json")
        if not summ:
            continue
        share = summ.get("dominant_share") or summ.get("dominant_expert_share") or summ
        if not isinstance(share, dict):
            continue
        share = {k: v for k, v in share.items() if isinstance(v, (int, float))}
        if not share:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(list(share.keys()), list(share.values()),
                      color=PAL[:len(share)])
        for b, v in zip(bars, share.values()):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                    ha="center", fontsize=9)
        ax.set_ylabel("dominant-expert share")
        ax.set_title("Expert routing"); plt.xticks(rotation=15)
        savefig(fig, f"{out}/07_routing/routing_share.png")


# ---------------------------------------------------------------- external
def make_external(runs, out):
    tags = ["deepdrid_teacher", "idrid_teacher"]
    acc, qwk, names = [], [], []
    for t in tags:
        o, _ = find_report(runs, t)
        if o:
            acc.append(o["accuracy"]); qwk.append(o["qwk"])
            names.append(TAGS[t][1])
    if not names:
        return
    x = range(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar([i - w/2 for i in x], acc, w, label="Accuracy", color=PAL[0])
    ax.bar([i + w/2 for i in x], qwk, w, label="QWK", color=PAL[1])
    for i, v in enumerate(acc):
        ax.text(i - w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(qwk):
        ax.text(i + w/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("External test-only (unseen datasets)"); ax.legend()
    savefig(fig, f"{out}/08_external/external_bars.png")


# ---------------------------------------------------------------- overall
def make_overall(runs, out):
    order = ["ddr_only", "aptos_only", "eyepacs_only", "teacher",
             "loeo_heldout_eyepacs", "loeo_heldout_ddr", "loeo_heldout_aptos",
             "labelfree", "nocausal", "student",
             "deepdrid_teacher", "idrid_teacher"]
    qwk, names = [], []
    for t in order:
        o, _ = find_report(runs, t)
        if o:
            qwk.append(o["qwk"]); names.append(TAGS[t][1])
    if not names:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, qwk, color=PAL[0])
    for b, v in zip(bars, qwk):
        ax.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=7)
    ax.set_ylim(0, 1.02); ax.set_ylabel("test QWK")
    ax.set_title("QWK across every run")
    plt.xticks(rotation=35, ha="right")
    savefig(fig, f"{out}/09_overall/qwk_all_runs.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    r, o = args.runs_dir, args.out_dir

    print("per-run curves");        make_curves(r, o)
    print("confusion matrices");    make_confusions(r, o)
    print("per-source bars");       make_persource_bars(r, o)
    print("teacher per-env");       make_perenv(r, o)
    print("LOEO summary");          make_loeo(r, o)
    print("ablation bars");         make_ablation(r, o)
    print("student trade-off");     make_student(r, o)
    print("calibration");           make_calibration(r, o)
    print("routing");               make_routing(r, o)
    print("external");              make_external(r, o)
    print("overall QWK");           make_overall(r, o)
    print(f"\nDone. Figures under {o}/ (stage-wise folders).")


if __name__ == "__main__":
    main()
