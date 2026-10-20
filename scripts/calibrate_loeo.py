"""Post-hoc temperature scaling for a leave-one-environment-out (LOEO) model.

Why a dedicated script (see REVIEWER_FIXES.md section 6):
`evaluate.py --calibrate-temperature` fits T on the validation split of the SAME
environments it then tests on. For a LOEO model that would leak the held-out
site into T. Here T is fitted ONLY on the validation split of the TRAINING
environments (default: ddr + aptos) and then applied, unchanged, to the
held-out site's test predictions that evaluate.py already saved.

No training is repeated: the script loads an existing `*_best.pt`, runs
inference on the (small) training-environment validation split, and re-uses
`preds_<tag>_test.npz`. Temperature scaling does not change argmax, so
accuracy / QWK stay identical; only ECE (and NLL) change.

Usage (teacher-style model, e.g. nocausal / full / labelfree):
    python scripts/calibrate_loeo.py \
        --config configs/ablation_nocausal.yaml \
        --ckpt runs/ablation_nocausal/nocausal_loeo_heldout_eyepacs_best.pt \
        --test-preds runs/ablation_nocausal/preds_nocausal_loeo_heldout_eyepacs_test.npz \
        --heldout eyepacs --out runs/ablation_nocausal/calib_nocausal_loeo_heldout_eyepacs.json

Plain ConvNeXt-Tiny baseline: add `--student`.
Label-free model: add `--quality-mode labelfree`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import TemperatureScaler, expected_calibration_error  # noqa: E402


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    p = _softmax(logits)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def calibrate_from_arrays(val_logits, val_labels, test_pred: dict, id2env: dict) -> dict:
    """Fit T on (val_logits, val_labels); report raw vs scaled ECE/NLL per test env."""
    ts = TemperatureScaler().fit(val_logits, val_labels)
    T = ts.T
    out = {"temperature": T, "n_val": int(len(val_labels)), "per_environment": {}}
    groups = {"overall": np.ones(len(test_pred["labels"]), dtype=bool)}
    for e in np.unique(test_pred["envs"]):
        groups[id2env[int(e)]] = test_pred["envs"] == e
    for name, m in groups.items():
        if m.sum() < 10:
            continue
        lg, lb = test_pred["logits"][m], test_pred["labels"][m]
        ece_raw, _ = expected_calibration_error(_softmax(lg), lb)
        ece_cal, _ = expected_calibration_error(_softmax(lg / T), lb)
        out["per_environment"][name] = {
            "n": int(m.sum()),
            "ece_raw": ece_raw, "ece_scaled": ece_cal,
            "nll_raw": _nll(lg, lb), "nll_scaled": _nll(lg / T, lb),
        }
    return out


def ENV2ID_NAME(heldout: str, id2env: dict) -> str:
    """ID2ENV maps id 1 to 'idrid' (last alias wins), so 'ddr' shows up as 'idrid'."""
    from src.data.datasets import ENV2ID
    return id2env[ENV2ID[heldout]]


def main():
    import torch
    from src.data.datasets import ID2ENV
    from src.engine.builders import build_loader, build_student, build_teacher
    from src.engine.evaluate import collect_predictions
    from src.utils import device_from, load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-preds", required=True, help="preds_<tag>_test.npz saved by evaluate.py")
    ap.add_argument("--heldout", default="eyepacs")
    ap.add_argument("--train-envs", nargs="*", default=None,
                    help="environments T is fitted on (default: every env except --heldout)")
    ap.add_argument("--student", action="store_true")
    ap.add_argument("--quality-mode", default=None, choices=[None, "calibrated", "labelfree"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_envs = args.train_envs or [e for e in ("eyepacs", "ddr", "aptos") if e != args.heldout]
    assert args.heldout not in train_envs, "held-out environment must not be used to fit T"

    cfg = load_config(args.config)
    device = device_from(cfg)
    model = build_student(cfg, device) if args.student else build_teacher(cfg, device, args.quality_mode)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)

    val_loader = build_loader(cfg, "val", train=False, envs=train_envs)
    val = collect_predictions(model, val_loader, device, is_student=args.student)
    test = dict(np.load(args.test_preds))

    res = calibrate_from_arrays(val["logits"], val["labels"], test, ID2ENV)
    res.update({"fitted_on_envs": train_envs, "heldout": args.heldout, "ckpt": args.ckpt})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)

    held_name = ENV2ID_NAME(args.heldout, ID2ENV)
    print(f"T = {res['temperature']:.3f}  (fitted on {train_envs} val, n={res['n_val']})")
    print(f"{'env':10s} {'n':>6s} {'ECE raw':>9s} {'ECE scaled':>11s} {'NLL raw':>9s} {'NLL scaled':>11s}")
    for name, r in res["per_environment"].items():
        tag = "  <- HELD-OUT" if name == held_name else ""
        print(f"{name:10s} {r['n']:6d} {r['ece_raw']:9.4f} {r['ece_scaled']:11.4f} "
              f"{r['nll_raw']:9.4f} {r['nll_scaled']:11.4f}{tag}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
