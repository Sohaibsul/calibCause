"""Inference + all four evaluation blocks of Figure 1."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from ..data.datasets import ID2ENV, NUM_GRADES
from ..utils import (TemperatureScaler, device_from, dump_json,
                     expected_calibration_error, get_logger, grading_metrics,
                     load_config)
from .builders import build_loader, build_student, build_teacher


@torch.no_grad()
def collect_predictions(model, loader, device, is_student: bool = False) -> dict:
    model.eval()
    logits, labels, envs, gates, quals = [], [], [], [], []
    for b in tqdm(loader, desc="infer", leave=False):
        x = b["image"].to(device, non_blocking=True)
        out = model(x)
        logits.append(out["logits"].float().cpu().numpy())
        labels.append(b["grade"].numpy())
        envs.append(b["env"].numpy())
        if not is_student:
            gates.append(out["gate_weights"].float().cpu().numpy())
            quals.append(out["quality_score"].float().cpu().numpy())
    res = {"logits": np.concatenate(logits), "labels": np.concatenate(labels),
           "envs": np.concatenate(envs)}
    if gates:
        res["gate_weights"] = np.concatenate(gates)
        res["quality_score"] = np.concatenate(quals)
    return res


def evaluate_predictions(pred: dict, per_env: bool = True, temperature: float | None = None):
    logits = pred["logits"] if temperature is None else pred["logits"] / temperature
    report = {"overall": grading_metrics(logits, pred["labels"], NUM_GRADES)}
    if per_env:
        report["per_environment"] = {}
        for e in np.unique(pred["envs"]):
            m = pred["envs"] == e
            if m.sum() < 10:
                continue
            report["per_environment"][ID2ENV[int(e)]] = grading_metrics(
                logits[m], pred["labels"][m], NUM_GRADES)
    if "quality_score" in pred:
        report["quality_strata"] = quality_stratified(logits, pred)
    return report


def quality_stratified(logits: np.ndarray, pred: dict, n_strata: int = 3):
    """Accuracy by predicted-quality tertile.

    This is the table that shows the model does not simply give up on the
    degraded images -- the quality-severity coupling the base paper documents
    means those strata carry more referable disease, not less.
    """
    q = pred["quality_score"]
    edges = np.quantile(q, np.linspace(0, 1, n_strata + 1))
    out = {}
    names = ["low_quality", "mid_quality", "high_quality"][:n_strata]
    for i, name in enumerate(names):
        upper = (q <= edges[i + 1]) if i == n_strata - 1 else (q < edges[i + 1])
        m = (q >= edges[i]) & upper
        if m.sum() < 10:
            continue
        stats = grading_metrics(logits[m], pred["labels"][m], NUM_GRADES)
        stats["n"] = int(m.sum())
        stats["referable_prevalence"] = float((pred["labels"][m] >= 2).mean())
        out[name] = stats
    return out


def fit_temperature(model, val_loader, device, is_student=False) -> float:
    val = collect_predictions(model, val_loader, device, is_student)
    ts = TemperatureScaler().fit(val["logits"], val["labels"])
    return ts.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--student", action="store_true")
    ap.add_argument("--quality-mode", default=None, choices=[None, "calibrated", "labelfree"])
    ap.add_argument("--calibrate-temperature", action="store_true")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    device = device_from(cfg)
    logger = get_logger("evaluate", cfg["out_dir"])

    model = build_student(cfg, device) if args.student else \
        build_teacher(cfg, device, args.quality_mode)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)

    T = None
    if args.calibrate_temperature:
        val_loader = build_loader(cfg, "val", train=False, envs=args.envs)
        val_pred = collect_predictions(model, val_loader, device, args.student)
        ts = TemperatureScaler().fit(val_pred["logits"], val_pred["labels"])
        T = ts.T
        np.savez(os.path.join(cfg["out_dir"], f"preds_{args.tag}_val.npz"), **val_pred)
        logger.info(f"fitted temperature T = {T:.3f} on the validation split")

    loader = build_loader(cfg, args.split, train=False, envs=args.envs)
    pred = collect_predictions(model, loader, device, args.student)
    report = evaluate_predictions(pred, temperature=T)
    if T is not None:
        report["temperature"] = T
        ece_raw, _ = expected_calibration_error(
            torch.softmax(torch.from_numpy(pred["logits"]), -1).numpy(), pred["labels"])
        report["ece_uncalibrated"] = ece_raw

    logger.info(f"{args.tag} | acc={report['overall']['accuracy']:.4f} "
                f"QWK={report['overall']['qwk']:.4f} ECE={report['overall']['ece']:.4f}")
    out_path = os.path.join(cfg["out_dir"], f"report_{args.tag}_{args.split}.json")
    dump_json(report, out_path)
    np.savez(os.path.join(cfg["out_dir"], f"preds_{args.tag}_{args.split}.npz"), **pred)
    logger.info(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
