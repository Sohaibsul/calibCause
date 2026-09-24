"""Turn everything under runs/ into paper-ready graphs and tables.

Scans every run directory for the artefacts the training and evaluation scripts
write, then produces:

  reports/curves_<tag>.png        training + validation curves for one run
  reports/confusion_<tag>.png     confusion matrix heatmap
  reports/comparison.png          accuracy / QWK / ECE bars across all runs
  reports/results_table.csv       one row per run  (open in Excel)
  reports/results_table.md        same table, markdown  (paste into notes)
  reports/results_table.tex       same table, LaTeX booktabs  (paste into paper)
  reports/per_environment.csv     per-run x per-environment breakdown
  reports/quality_strata.csv      accuracy per predicted-quality tertile
  reports/ablation_deltas.md      Module 3 / Module 1 / distillation deltas

Usage:
    python scripts/make_report.py                    # scans ./runs
    python scripts/make_report.py --runs runs --out reports
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Prolif."]
METRIC_COLS = ["accuracy", "qwk", "macro_f1", "auc_ovr", "referable_auc", "ece"]
PRETTY = {"accuracy": "Acc.", "qwk": "QWK", "macro_f1": "Macro-F1",
          "auc_ovr": "AUC", "referable_auc": "Ref. AUC", "ece": "ECE"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def find_histories(runs_dir: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(runs_dir, "**", "history_*.json"), recursive=True):
        tag = re.sub(r"^history_|\.json$", "", os.path.basename(p))
        out[tag] = load_json(p)
    return out


def find_reports(runs_dir: str) -> dict:
    out = {}
    for p in glob.glob(os.path.join(runs_dir, "**", "report_*.json"), recursive=True):
        name = re.sub(r"^report_|\.json$", "", os.path.basename(p))
        out[name] = load_json(p)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_curves(tag: str, hist: dict, out_dir: str):
    rows = hist.get("history", [])
    if not rows:
        return None
    ep = [r["epoch"] for r in rows]
    tr = [r.get("train", {}) for r in rows]
    va = [r.get("val", {}) for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # loss / distillation terms
    if "loss" in tr[0]:
        ax[0].plot(ep, [t.get("loss") for t in tr], marker="o", label="total")
    if "cls" in tr[0]:
        ax[0].plot(ep, [t.get("cls") for t in tr], marker=".", label="classification")
    if "env" in tr[0] and any(t.get("env", 0) for t in tr):
        ax[0].plot(ep, [t.get("env") for t in tr], marker=".", label="environment (adv.)")
    if "soft" in tr[0]:
        ax[0].plot(ep, [t.get("soft") for t in tr], marker=".", label="KD soft")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title(f"{tag}: training loss"); ax[0].legend(); ax[0].grid(alpha=0.3)

    # accuracy
    if "acc" in tr[0]:
        ax[1].plot(ep, [t.get("acc") for t in tr], marker="o", label="train")
    ax[1].plot(ep, [v.get("accuracy") for v in va], marker="s", label="val")
    if "agree" in tr[0]:
        ax[1].plot(ep, [t.get("agree") for t in tr], marker=".", label="teacher agreement")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
    ax[1].set_title(f"{tag}: accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)

    # QWK + ECE (model selection metric and calibration)
    ax[2].plot(ep, [v.get("qwk") for v in va], marker="s", color="tab:green", label="val QWK")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("QWK", color="tab:green")
    ax2 = ax[2].twinx()
    ax2.plot(ep, [v.get("ece") for v in va], marker=".", color="tab:red", label="val ECE")
    ax2.set_ylabel("ECE", color="tab:red")
    ax[2].set_title(f"{tag}: QWK / calibration"); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"curves_{tag}.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_confusion(name: str, cm: list, out_dir: str):
    cm = np.asarray(cm, dtype=float)
    if cm.size == 0:
        return None
    norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(GRADES)), GRADES, rotation=45, ha="right")
    ax.set_yticks(range(len(GRADES)), GRADES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}: confusion (row-normalised)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, f"confusion_{name}.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_comparison(df: pd.DataFrame, out_dir: str):
    if df.empty:
        return None
    metrics = [m for m in ("accuracy", "qwk", "ece") if m in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        d = df.dropna(subset=[m]).sort_values(m, ascending=(m == "ece"))
        ax.barh(d["run"], d[m], color="tab:blue" if m != "ece" else "tab:red")
        ax.set_xlabel(PRETTY.get(m, m))
        ax.set_title(("lower is better" if m == "ece" else "higher is better"))
        for i, v in enumerate(d[m]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def build_main_table(reports: dict) -> pd.DataFrame:
    rows = []
    for name, rep in sorted(reports.items()):
        ov = rep.get("overall", {})
        row = {"run": name}
        row.update({k: ov.get(k) for k in METRIC_COLS})
        if "temperature" in rep:
            row["temperature"] = rep["temperature"]
            row["ece_uncal"] = rep.get("ece_uncalibrated")
        rows.append(row)
    return pd.DataFrame(rows)


def build_env_table(reports: dict) -> pd.DataFrame:
    rows = []
    for name, rep in sorted(reports.items()):
        for env, stats in (rep.get("per_environment") or {}).items():
            rows.append({"run": name, "environment": env,
                         **{k: stats.get(k) for k in METRIC_COLS}})
    return pd.DataFrame(rows)


def build_quality_table(reports: dict) -> pd.DataFrame:
    rows = []
    for name, rep in sorted(reports.items()):
        for stratum, stats in (rep.get("quality_strata") or {}).items():
            rows.append({"run": name, "stratum": stratum, "n": stats.get("n"),
                         "referable_prevalence": stats.get("referable_prevalence"),
                         **{k: stats.get(k) for k in ("accuracy", "qwk", "ece")}})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, float_fmt: str = "%.4f") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "--" if pd.isna(v) else float_fmt % v)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "--" if pd.isna(v) else f"{v:.4f}")
    cols = "l" + "r" * (len(d.columns) - 1)
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(str(c).replace("_", r"\_") for c in d.columns) + r" \\",
             r"\midrule"]
    for r in d.itertuples(index=False):
        lines.append(" & ".join(str(v).replace("_", r"\_") for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def ablation_deltas(main: pd.DataFrame, histories: dict) -> str:
    """The three numbers Figure 1's evaluation block actually asks for."""
    def get(run_substr, metric):
        m = main[main["run"].str.contains(run_substr, case=False, na=False)]
        return None if m.empty else m.iloc[0][metric]

    out = ["# Ablation deltas", ""]

    t_acc, t_qwk = get("teacher", "accuracy"), get("teacher", "qwk")
    lf_acc, lf_qwk = get("labelfree", "accuracy"), get("labelfree", "qwk")
    if t_acc is not None and lf_acc is not None:
        out += ["## Module 3 -- cost of removing human quality labels",
                f"- accuracy: {t_acc:.4f} (calibrated) vs {lf_acc:.4f} (label-free) "
                f"-> delta {t_acc - lf_acc:+.4f}",
                f"- QWK: {t_qwk:.4f} vs {lf_qwk:.4f} -> delta {t_qwk - lf_qwk:+.4f}", ""]

    nc_acc, nc_qwk = get("nocausal", "accuracy"), get("nocausal", "qwk")
    if t_acc is not None and nc_acc is not None:
        out += ["## Module 1 -- gain from the environment-invariant causal block",
                f"- accuracy: {t_acc:.4f} (with) vs {nc_acc:.4f} (without) "
                f"-> delta {t_acc - nc_acc:+.4f}",
                f"- QWK: {t_qwk:.4f} vs {nc_qwk:.4f} -> delta {t_qwk - nc_qwk:+.4f}",
                "", "Read this on the *cross-environment* runs, not in-domain: the block is",
                "meant to trade a little in-domain fit for held-out-site robustness.", ""]

    s_acc, s_qwk = get("student", "accuracy"), get("student", "qwk")
    kd = next((h for k, h in histories.items() if "student" in k), None)
    if s_acc is not None and t_acc is not None:
        out += ["## Module 2 -- distillation trade-off",
                f"- accuracy: {t_acc:.4f} (teacher) vs {s_acc:.4f} (student) "
                f"-> delta {s_acc - t_acc:+.4f}",
                f"- QWK: {t_qwk:.4f} vs {s_qwk:.4f} -> delta {s_qwk - t_qwk:+.4f}"]
        if kd and "param_ratio" in kd:
            out += [f"- parameters: {kd['teacher_params']/1e6:.2f}M -> "
                    f"{kd['student_params']/1e6:.2f}M ({kd['param_ratio']*100:.1f}%)"]
        out += [""]

    loeo = main[main["run"].str.contains("loeo", case=False, na=False)]
    if not loeo.empty:
        out += ["## Cross-environment (leave-one-environment-out)", "",
                to_markdown(loeo[["run", "accuracy", "qwk", "ece"]]), ""]

    if len(out) == 2:
        out += ["_No comparable runs found yet -- train the teacher first._"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    histories = find_histories(args.runs)
    reports = find_reports(args.runs)

    if not histories and not reports:
        raise SystemExit(f"nothing found under {args.runs}/ -- train something first")

    made = []
    for tag, hist in histories.items():
        p = plot_curves(tag, hist, args.out)
        if p:
            made.append(p)

    for name, rep in reports.items():
        cm = rep.get("overall", {}).get("confusion_matrix")
        if cm:
            p = plot_confusion(name, cm, args.out)
            if p:
                made.append(p)

    main_tbl = build_main_table(reports)
    if not main_tbl.empty:
        main_tbl.to_csv(os.path.join(args.out, "results_table.csv"), index=False)
        with open(os.path.join(args.out, "results_table.md"), "w") as f:
            f.write("# Main results\n\n" + to_markdown(main_tbl) + "\n")
        with open(os.path.join(args.out, "results_table.tex"), "w") as f:
            f.write(to_latex(main_tbl,
                             "Grading performance of every trained configuration.",
                             "tab:main_results"))
        p = plot_comparison(main_tbl, args.out)
        if p:
            made.append(p)

    env_tbl = build_env_table(reports)
    if not env_tbl.empty:
        env_tbl.to_csv(os.path.join(args.out, "per_environment.csv"), index=False)
        with open(os.path.join(args.out, "per_environment.md"), "w") as f:
            f.write("# Per-environment breakdown\n\n" + to_markdown(env_tbl) + "\n")

    q_tbl = build_quality_table(reports)
    if not q_tbl.empty:
        q_tbl.to_csv(os.path.join(args.out, "quality_strata.csv"), index=False)
        with open(os.path.join(args.out, "quality_strata.md"), "w") as f:
            f.write("# Accuracy by predicted-quality tertile\n\n"
                    "`referable_prevalence` is the share of grade>=2 cases in that "
                    "stratum -- the quality-severity coupling the base paper documents.\n\n"
                    + to_markdown(q_tbl) + "\n")

    with open(os.path.join(args.out, "ablation_deltas.md"), "w") as f:
        f.write(ablation_deltas(main_tbl, histories))

    print(f"runs found: {len(histories)} histories, {len(reports)} reports")
    if not main_tbl.empty:
        print("\n" + to_markdown(main_tbl))
    print(f"\nwrote {len(made)} figures + tables to {args.out}/")


if __name__ == "__main__":
    main()
