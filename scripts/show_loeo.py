"""LOEO summary table: held-out QWK per model per fold (test split, held-out site only)."""
import json, os

MODELS = {
    "Full (CalibCause)": "runs/stage2/report_loeo_heldout_{e}_test.json",
    "Balanced-gentle":   "runs/stage2_balanced_gentle/report_loeo_heldout_{e}_test.json",
    "No-causal":         "runs/ablation_nocausal/report_nocausal_loeo_heldout_{e}_test.json",
    "Label-free":        "runs/ablation_labelfree/report_labelfree_loeo_heldout_{e}_test.json",
    "ConvNeXt-T plain (fixed)":        "runs/baseline/report_baseline2_loeo_heldout_{e}_test.json",
    "ConvNeXt-T OLD recipe (invalid)": "runs/baseline/report_baseline_loeo_heldout_{e}_test.json",
}
FOLDS = ["eyepacs", "ddr", "aptos"]
ALIASES = {"ddr": ["ddr", "idrid"], "eyepacs": ["eyepacs"], "aptos": ["aptos"]}

def get(path, fold):
    if not os.path.exists(path):
        return None
    pe = json.load(open(path))["per_environment"]
    for k in ALIASES[fold]:
        if k in pe:
            return pe[k]
    return None

print(f"{'model':34s}" + "".join(f"{f:>10s}" for f in FOLDS) + f"{'mean':>10s}")
for name, tmpl in MODELS.items():
    vals = []
    for f in FOLDS:
        e = get(tmpl.format(e=f), f)
        vals.append(e["qwk"] if e else None)
    got = [v for v in vals if v is not None]
    mean = f"{sum(got)/len(got):.4f}" if len(got) == len(FOLDS) else "--"
    print(f"{name:34s}" + "".join(f"{(f'{v:.4f}' if v is not None else '--'):>10s}" for v in vals) + f"{mean:>10s}")
