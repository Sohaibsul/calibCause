"""Stage 2a + Module 1 -- quality-gated MoE teacher with environment-invariant
adversarial training.

Run (in-domain, calibrated quality, full model):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml

Run (Module 3 ablation, label-free gating):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
        --quality-mode labelfree --tag labelfree

Run (cross-environment, leave-one-environment-out):
    python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
        --train-envs eyepacs ddr --test-envs aptos --tag loeo_aptos
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..losses import OrdinalAwareCE, entropy_regularizer
from ..models import dann_lambda
from ..utils import (AverageMeter, device_from, dump_json, get_logger,
                     load_config, save_checkpoint, set_seed)
from .builders import build_loader, build_teacher, class_weights_from
from .evaluate import collect_predictions, evaluate_predictions


def run_epoch(model, loader, opt, sched, scaler, crit, cfg, device, epoch,
              total_epochs, logger):
    model.train()
    meters = {k: AverageMeter() for k in ("loss", "cls", "env", "ent", "acc")}
    steps_per_epoch = max(1, len(loader))
    total_steps = total_epochs * steps_per_epoch
    lam_max = cfg["train"].get("env_lambda", 1.0)
    w_env = cfg["train"].get("env_weight", 0.3)
    w_ent = cfg["train"].get("gate_entropy_weight", 0.05)
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
            loss = l_cls + w_env * l_env + w_ent * l_ent

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
                     ("env", float(l_env)), ("ent", l_ent.item()), ("acc", acc)):
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
    ap.add_argument("--resume", default=None)
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
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False)["model"], strict=False)

    cw = class_weights_from(train_loader).to(device) if cfg["train"].get("class_weighted_loss") \
        else None
    crit = OrdinalAwareCE(smoothing=cfg["train"].get("label_smoothing", 0.05),
                          ordinal_weight=cfg["train"].get("ordinal_weight", 0.5),
                          class_weights=cw).to(device)

    epochs = cfg["train"]["epochs"]
    opt = torch.optim.AdamW(model.trainable_groups(cfg["train"]["lr"],
                                                   cfg["train"].get("lr_backbone_mult", 0.1)),
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    best, history = -1.0, []
    ckpt_path = os.path.join(cfg["out_dir"], f"{args.tag}_best.pt")
    for ep in range(epochs):
        stats = run_epoch(model, train_loader, opt, sched, scaler, crit, cfg,
                          device, ep, epochs, logger)
        pred = collect_predictions(model, val_loader, device)
        rep = evaluate_predictions(pred, per_env=True)
        score = rep["overall"]["qwk"]  # model selection on QWK, as in the base paper
        history.append({"epoch": ep + 1, "train": stats, "val": rep["overall"]})
        logger.info(f"  val acc={rep['overall']['accuracy']:.4f} qwk={score:.4f} "
                    f"ece={rep['overall']['ece']:.4f}")
        if score > best:
            best = score
            save_checkpoint({"model": model.state_dict(), "cfg": cfg, "epoch": ep,
                             "val": rep, "quality_mode": args.quality_mode}, ckpt_path)
            logger.info(f"  new best QWK -> saved {ckpt_path}")
    dump_json({"best_val_qwk": best, "history": history},
              os.path.join(cfg["out_dir"], f"history_{args.tag}.json"))


if __name__ == "__main__":
    main()
