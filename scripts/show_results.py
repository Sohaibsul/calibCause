"""Quick status + final held-out-EyePACS comparison (reviewer comments 1, 2, 5).
Read-only: never touches checkpoints or running jobs."""
import glob, json, os
import pandas as pd

print("=== per-epoch progress (val on the TRAINING envs, NOT the held-out result) ===")
for f in sorted(glob.glob("runs/**/epoch_metrics_*eyepacs*.csv", recursive=True)):
    d = pd.read_csv(f)
    last = d.iloc[-1]
    print(f"{f}\n   epochs done: {int(last['epoch'])}  last QWK: {last['qwk']:.4f}  "
          f"best QWK: {d['qwk'].max():.4f} (epoch {int(d.loc[d['qwk'].idxmax(), 'epoch'])})")

print("\n=== FINAL held-out EyePACS (test split, unseen site) ===")
rows = [
    ("Full model (original config)",  "runs/stage2/report_loeo_heldout_eyepacs_test.json"),
    ("Routing-fixed (balanced_gentle)", "runs/stage2_balanced_gentle/report_loeo_heldout_eyepacs_test.json"),
    ("No-causal ablation",            "runs/ablation_nocausal/report_nocausal_loeo_heldout_eyepacs_test.json"),
    ("Label-free ablation",           "runs/ablation_labelfree/report_labelfree_loeo_heldout_eyepacs_test.json"),
    ("Plain ConvNeXt-Tiny baseline",  "runs/baseline/report_baseline_loeo_heldout_eyepacs_test.json"),
]
print(f"{'model':36s} {'QWK':>7s} {'acc':>7s} {'bal_acc':>8s} {'ref_AUC':>8s}")
for name, p in rows:
    if not os.path.exists(p):
        print(f"{name:36s}   (not finished yet)")
        continue
    m = json.load(open(p))["per_environment"]["eyepacs"]
    print(f"{name:36s} {m['qwk']:7.4f} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} {m['referable_auc']:8.4f}")
print("\nSingle seed: treat QWK differences below ~0.02 as noise.")
