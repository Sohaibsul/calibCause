"""Stage 2a + Module 1 -- quality-gated MoE teacher with environment-invariant
adversarial training.

Run (in-domain, calibrated quality, full model):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml

Run (Module 3 ablation, label-free gating):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
        --quality-mode labelfree --tag labelfree

Run (cross-environment, leave-one-environment-out):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
        --train-envs eyepacs ddr --val-envs eyepacs ddr --tag loeo_heldout_aptos
    (held-out score: evaluate.py --split test, read per_environment[<env>])
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..losses import OrdinalAwareCE, entropy_regularizer, load_balance_loss
from ..models import dann_lambda
from ..utils import (AverageMeter, device_from, dump_json, get_logger,
                     load_config, save_checkpoint, set_seed)
from .builders import build_loader, build_teacher, class_weights_from
from .evaluate import collect_predictions, evaluate_predictions


def run_epoch(model, loader, opt, sched, scaler, crit, cfg, device, epoch,
              total_epochs, logger):
    model.train()
    meters = {k: AverageMeter() for k in ("loss", "cls", "env", "ent", "bal", "acc")}
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    lam_max = cfg["train"].get("env_lambda", 1.0)
    w_env = cfg["train"].get("env_weight", 0.3)
    w_ent = cfg["train"].get("gate_entropy_weight", 0.05)
    # dataset-wide MoE load-balancing (reviewer fix: vascular expert routed to
    # ~0% usage even though per-sample entropy_regularizer was satisfied).
    # Off by default (0.0) so existing runs/configs are reproduced exactly;
    # opt in per-config with train.load_balance_weight.
    w_bal = cfg["train"].get("load_balance_weight", 0.0)
    warmup_ep = cfg["train"].get("adv_warmup_epochs", 1)

    pbar = tqdm(loader, desc=f"[stage2] epoch {epoch + 1}/{total_epochs}")
    for it, b in enumerate(pbar):
        step = epoch * steps_per_epoch + it
        # the discriminator is disabled for the first epoch(s): an adversary
        # attached to a representation that carries no disease signal yet just
        # injects noise (Ganin et al. schedule + explicit warm-up)
        lam = 0.0 if epoch < warmup_ep else dann_lambda(step, total_steps, lam_max)

        x = b["image"].to(device, non_blocking=True)
        y = b["grade"].to(device, non_blocking=True)
        e = b["env"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(x, grl_lambda=lam)
            l_cls = crit(out["logits"], y)
            l_ent = entropy_regularizer(out["gate_weights"],
                                        cfg["train"].get("gate_target_entropy", 0.6))
            l_env = torch.zeros((), device=device)
            if "env_logits" in out and lam > 0:
                l_env = F.cross_entropy(out["env_logits"], e)
            l_bal = load_balance_loss(out["gate_weights"]) if w_bal > 0 \
                else torch.zeros((), device=device)
            loss = l_cls + w_env * l_env + w_ent * l_ent + w_bal * l_bal

        scaler.scale(loss).backward()
        if cfg["train"].get("clip_grad", 0):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_grad"])
        scaler.step(opt)
        scaler.update()
        sched.step()   # OneCycleLR is a per-batch schedule

        n = x.size(0)
        acc = (out["logits"].argmax(-1) == y).float().mean().item()
        for k, v in (("loss", loss.item()), ("cls", l_cls.item()),
                     ("env", float(l_env)), ("ent", l_ent.item()),
                     ("bal", float(l_bal)), ("acc", acc)):
            meters[k].update(v, n)
        pbar.set_postfix(loss=f"{meters['loss'].avg:.3f}", acc=f"{meters['acc'].avg:.3f}",
                         lam=f"{lam:.2f}")

    logger.info(f"epoch {epoch + 1}: " +
                " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items()))
    return {k: m.avg for k, m in meters.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quality-mode", default=None, choices=["calibrated", "labelfree"],
                    help="Module 3 ablation switch")
    ap.add_argument("--train-envs", nargs="*", default=None)
    ap.add_argument("--val-envs", nargs="*", default=None)
    ap.add_argument("--no-causal", action="store_true", help="Module 1 ablation")
    ap.add_argument("--tag", default="teacher")
    ap.add_argument("--resume", default=None,
                    help="path to a checkpoint. If it has optimizer/scheduler/epoch state "
                         "(saved as <out_dir>/<tag>_last.pt by this script), training truly "
                         "continues from that epoch. If it's a weights-only checkpoint (e.g. "
                         "a _best.pt from a different run), only the model weights are loaded "
                         "and training restarts at epoch 0 with a fresh LR schedule.")
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    if args.no_causal:
        cfg["model"]["use_causal"] = False
        cfg["model"]["use_env_disc"] = False
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = device_from(cfg)
    logger = get_logger(f"stage2_{args.tag}", cfg["out_dir"])
    logger.info(f"device={device} | tag={args.tag} | train_envs={args.train_envs}")

    train_loader = build_loader(cfg, "train", train=True, envs=args.train_envs)
    val_loader = build_loader(cfg, "val", train=False, envs=args.val_envs or args.train_envs)
    logger.info(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_teacher(cfg, device, args.quality_mode)

    epochs = cfg["train"]["epochs"]
    opt = torch.optim.AdamW(model.trainable_groups(cfg["train"]["lr"],
                                                   cfg["train"].get("lr_backbone_mult", 0.1)),
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    best, history, start_ep = -1.0, [], 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        if "optimizer" in ck and "scheduler" in ck:
            # true resume: optimizer momentum, OneCycleLR's internal step
            # count, epoch and best-so-far all carried over exactly, so
            # training continues as if it had never stopped
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            best = ck.get("best", -1.0)
            history = ck.get("history", [])
            start_ep = ck["epoch"] + 1
            logger.info(f"resuming from epoch {start_ep + 1} "
                       f"(checkpoint was saved after epoch {ck['epoch'] + 1}, best qwk so far={best:.4f})")
        else:
            logger.info(f"loaded weights only from {args.resume} (no optimizer/scheduler state "
                       f"in this checkpoint) -- restarting at epoch 1 with a fresh LR schedule")

    cw = class_weights_from(train_loader).to(device) if cfg["train"].get("class_weighted_loss") \
        else None
    crit = OrdinalAwareCE(smoothing=cfg["train"].get("label_smoothing", 0.05),
                          ordinal_weight=cfg["train"].get("ordinal_weight", 0.5),
                          class_weights=cw).to(device)

    best_path = os.path.join(cfg["out_dir"], f"{args.tag}_best.pt")
    last_path = os.path.join(cfg["out_dir"], f"{args.tag}_last.pt")
    history_path = os.path.join(cfg["out_dir"], f"history_{args.tag}.json")
    metrics_csv_path = os.path.join(cfg["out_dir"], f"epoch_metrics_{args.tag}.csv")
    # scalar (flattenable) fields from grading_metrics() -- everything except
    # the nested per_class / confusion_matrix / referable_counts dicts, which
    # stay in the JSON history but don't fit a flat CSV row
    _csv_fields = ["accuracy", "balanced_accuracy", "qwk", "macro_f1", "macro_precision",
                  "macro_sensitivity", "auc_ovr", "referable_auc", "referable_sensitivity",
                  "referable_specificity", "referable_precision", "referable_f1", "ece"]

    for ep in range(start_ep, epochs):
        stats = run_epoch(model, train_loader, opt, sched, scaler, crit, cfg,
                          device, ep, epochs, logger)
        pred = collect_predictions(model, val_loader, device)
        # per_env=True so per-environment (e.g. ddr / aptos) breakdowns are
        # computed every epoch too, not just at the final evaluate.py pass --
        # useful if training gets interrupted and you want to see how it was
        # actually generalising across environments up to that point
        rep = evaluate_predictions(pred, per_env=True)
        ov = rep["overall"]
        score = ov["qwk"]  # model selection on QWK, as in the base paper
        history.append({"epoch": ep + 1, "train": stats, "val": rep})
        logger.info(f"  val acc={ov['accuracy']:.4f} bal_acc={ov['balanced_accuracy']:.4f} "
                    f"qwk={score:.4f} macro_f1={ov['macro_f1']:.4f} "
                    f"referable_auc={ov['referable_auc']:.4f} "
                    f"referable_sens={ov['referable_sensitivity']:.4f} ece={ov['ece']:.4f}")
        if score > best:
            best = score
            save_checkpoint({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                             "val": rep, "quality_mode": args.quality_mode}, best_path)
            logger.info(f"  new best QWK -> saved {best_path}")
        # saved every epoch (not just on new-best) so a crash mid-run loses at
        # most one epoch of progress, and carries full optimizer/scheduler
        # state so --resume can continue the LR schedule exactly, not restart it
        save_checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(), "cfg": cfg, "epoch": ep,
                         "best": best, "history": history, "val": rep,
                         "quality_mode": args.quality_mode}, last_path)
        # written every epoch (not just at the very end) so a crash doesn't
        # lose the whole run's metric history -- this was previously only
        # dumped after the final epoch, meaning an interrupted run left NO
        # metrics file behind at all
        dump_json({"best_val_qwk": best, "history": history}, history_path)

        write_header = not os.path.exists(metrics_csv_path)
        with open(metrics_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch"] + _csv_fields)
            if write_header:
                w.writeheader()
            w.writerow({"epoch": ep + 1, **{k: ov.get(k) for k in _csv_fields}})


if __name__ == "__main__":
    main()
