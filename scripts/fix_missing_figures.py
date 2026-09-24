"""Generate the two figures that make_paper_figures.py missed:
   confusion matrices (bug: key was nested under "overall")
   calibration summary bars (JSON only has flat summary numbers, not per-bin data)

Usage:
    python scripts/fix_missing_figures.py --runs-dir runs --out-dir figures
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = ["#4C72B0", "#DD8452", "#55A868", "#8172B2", "#C44E52"]
GRADES = ["0", "1", "2", "3", "4"]
TAGS = {
    "ddr_only": ("01_per_source", "DDR-only"),
    "aptos_only": ("01_per_source", "APTOS-only"),
    "eyepacs_only": ("01_per_source", "EyePACS-only"),
    "teacher": ("02_combined", "Combined (teacher)"),
    "loeo_heldout_eyepacs": ("03_loeo", "Held-out EyePACS"),
    "loeo_heldout_ddr": ("03_loeo", "Held-out DDR"),
    "loeo_heldout_aptos": ("03_loeo", "Held-out APTOS"),
    "labelfree": ("04_ablations", "Label-free"),
    "nocausal": ("04_ablations", "No-causal"),
    "student": ("05_student", "Student"),
    "deepdrid_teacher": ("08_external", "DeepDRiD"),
    "idrid_teacher": ("08_external", "IDRiD"),
}

def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print("  wrote", path)

def find_report(runs, tag):
    for p in glob.glob(f"{runs}/**/report_{tag}_*.json", recursive=True):
        d = json.load(open(p))
        return d.get("overall", d)   # FIX: confusion_matrix lives inside "overall"
    return None

def confusions(runs, out):
    for tag, (folder, label) in TAGS.items():
        o = find_report(runs, tag)
        if not o or "confusion_matrix" not in o:
            continue
        cm = np.array(o["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(5)); ax.set_xticklabels(GRADES)
        ax.set_yticks(range(5)); ax.set_yticklabels(GRADES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion \u2014 {label}")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        save(fig, f"{out}/{folder}/confusion_{tag}.png")

def calibration_bars(runs, out):
    paths = glob.glob(f"{runs}/**/uncertainty_summary.json", recursive=True)
    if not paths:
        print("  [skip] no uncertainty_summary.json found"); return
    s = json.load(open(paths[0]))

    # Panel 1: ECE before vs after temperature scaling
    fig, ax = plt.subplots(figsize=(5, 4.5))
    vals = [s["ece_uncalibrated"], s["ece_calibrated"]]
    bars = ax.bar(["Uncalibrated", "Temp-scaled"], vals, color=[PAL[4], PAL[0]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("ECE"); ax.set_title(f"Calibration (T={s['temperature']:.3f})")
    save(fig, f"{out}/06_calibration/reliability.png")

    # Panel 2: accuracy vs coverage (deferral)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cov = [1.0, 0.9, 0.8]
    acc = [s["accuracy"], s["accuracy_at_90pct_coverage"], s["accuracy_at_80pct_coverage"]]
    ax.plot(cov, acc, marker="o", color=PAL[2])
    for c, a in zip(cov, acc):
        ax.text(c, a + 0.003, f"{a:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("coverage"); ax.set_ylabel("accuracy on retained set")
    ax.set_title("Accuracy vs.\\ coverage (deferral)")
    ax.invert_xaxis()
    save(fig, f"{out}/06_calibration/deferral.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    print("confusion matrices"); confusions(args.runs_dir, args.out_dir)
    print("calibration bars");   calibration_bars(args.runs_dir, args.out_dir)

if __name__ == "__main__":
    main()
