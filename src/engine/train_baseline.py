"""Plain ConvNeXt-Tiny baseline (reviewer comment 2).

"Have you tried ConvNeXt-Tiny trained with the same split? If it works about
the same (hopefully it wouldn't) that would call into question the benefit
of the new pipeline."

This reuses the exact same backbone class already in the codebase
(`SingleStreamStudent`, otherwise used as the Module-2 distillation student)
as a standalone classifier -- no quality branch, no MoE gate/experts, no
causal transform, no adversarial environment discriminator, no distillation.
It is trained with the same ordinal-aware loss, optimizer, LR schedule and
epoch budget as the full model (`train_stage2.py`), so a comparison against
Table 1 isolates the effect of the architecture rather than a confound in
training recipe or backbone choice.

Run (per-source, mirrors the full model's protocol):
    python -m src.engine.train_baseline --config configs/baseline.yaml \
        --train-envs eyepacs --val-envs eyepacs --tag baseline_eyepacs

Run (combined):
    python -m src.engine.train_baseline --config configs/baseline.yaml \
        --tag baseline_combined

Run (LOEO, one call per held-out environment):
    python -m src.engine.train_baseline --config configs/baseline.yaml \
        --train-envs ddr aptos --val-envs ddr aptos --tag baseline_loeo_heldout_eyepacs

Evaluate exactly like the other models (the `--student` flag on evaluate.py
is for this same single-logits-output shape):
    python -m src.engine.evaluate --config configs/baseline.yaml \
        --ckpt runs/baseline/baseline_eyepacs_best.pt --student \
        --split test --envs eyepacs --tag baseline_eyepacs
"""
from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from ..losses import OrdinalAwareCE
from ..utils import AverageMeter, device_from, dump_json, get_logger, load_config, save_checkpoint, set_seed
from .builders import build_loader, build_student, class_weights_from
from .evaluate import collect_predictions, evaluate_predictions


def run_epoch(model, loader, opt, sched, scaler, crit, device, epoch, total_epochs, logger, clip_grad=0.0):
    model.train()
    meters = {k: AverageMeter() for k in ("loss", "acc")}
    pbar = tqdm(loader, desc=f"[baseline] epoch {epoch + 1}/{total_epochs}")
    for b in pbar:
        x = b["image"].to(device, non_blocking=True)
        y = b["grade"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            out = model(x)
            loss = crit(out["logits"], y)

        scaler.scale(loss).backward()
        if clip_grad:
            # same as train_stage2.py: unscale first, so the threshold is in true gradient units
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(opt)
        scaler.update()
        sched.step()

        n = x.size(0)
        acc = (out["logits"].argmax(-1) == y).float().mean().item()
        meters["loss"].update(loss.item(), n)
        meters["acc"].update(acc, n)
        pbar.set_postfix(loss=f"{meters['loss'].avg:.3f}", acc=f"{meters['acc'].avg:.3f}")

    logger.info(f"epoch {epoch + 1}: " + " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items()))
    return {k: m.avg for k, m in meters.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-envs", nargs="*", default=None)
    ap.add_argument("--val-envs", nargs="*", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = device_from(cfg)
    logger = get_logger(f"baseline_{args.tag}", cfg["out_dir"])
    logger.info(f"device={device} | tag={args.tag} | train_envs={args.train_envs} "
               f"(plain {cfg['student']['backbone']}, no quality/gate/causal/KD)")

    train_loader = build_loader(cfg, "train", train=True, envs=args.train_envs)
    val_loader = build_loader(cfg, "val", train=False, envs=args.val_envs or args.train_envs)
    logger.info(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_student(cfg, device)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    cw = class_weights_from(train_loader).to(device) if cfg["train"].get("class_weighted_loss") else None
    crit = OrdinalAwareCE(smoothing=cfg["train"].get("label_smoothing", 0.05),
                          ordinal_weight=cfg["train"].get("ordinal_weight", 0.5),
                          class_weights=cw).to(device)

    epochs = cfg["train"]["epochs"]
    # Same optimizer recipe as train_stage2.py: the pretrained backbone ("body") runs at
    # lr * lr_backbone_mult, the new classification head at the full lr. (A single LR for
    # everything made the backbone 10x too fast and the run diverged.)
    mult = cfg["train"].get("lr_backbone_mult", 0.1)
    body = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("body.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("body.")]
    assert body and head, "expected both 'body.*' and head parameters"
    opt = torch.optim.AdamW([{"params": body, "lr": cfg["train"]["lr"] * mult},
                             {"params": head, "lr": cfg["train"]["lr"]}],
                            weight_decay=cfg["train"]["weight_decay"])
    logger.info(f"optimizer groups: body lr={cfg['train']['lr'] * mult:g} ({len(body)} tensors) | "
                f"head lr={cfg['train']['lr']:g} ({len(head)} tensors) | "
                f"clip_grad={cfg['train'].get('clip_grad', 0.0)}")
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups], epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    best, history, start_ep = -1.0, [], 0
    ckpt_path = os.path.join(cfg["out_dir"], f"{args.tag}_best.pt")
    last_path = os.path.join(cfg["out_dir"], f"{args.tag}_last.pt")
    # A `_last.pt` (written after every epoch) carries optimizer + LR scheduler +
    # AMP scaler + history, so a power cut / reboot loses at most one epoch.
    # A weights-only checkpoint (old `_best.pt`) still loads, but restarts the LR schedule.
    if ck is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        logger.info(f"resuming from epoch {start_ep + 1} (checkpoint saved after epoch {ck['epoch'] + 1}, "
                    f"best qwk so far={best:.4f})")
    for ep in range(start_ep, epochs):
        stats = run_epoch(model, train_loader, opt, sched, scaler, crit, device, ep, epochs, logger,
                          clip_grad=cfg["train"].get("clip_grad", 0.0))
        pred = collect_predictions(model, val_loader, device, is_student=True)
        rep = evaluate_predictions(pred, per_env=True)
        score = rep["overall"]["qwk"]
        history.append({"epoch": ep + 1, "train": stats, "val": rep["overall"]})
        logger.info(f"  val acc={rep['overall']['accuracy']:.4f} qwk={score:.4f} "
                    f"ece={rep['overall']['ece']:.4f}")
        if score > best:
            best = score
            save_checkpoint({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val": rep}, ckpt_path)
            logger.info(f"  new best QWK -> saved {ckpt_path}")
        save_checkpoint({"model": model.state_dict(), "optimizer": opt.state_dict(),
                         "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                         "epoch": ep, "best": best, "history": history, "cfg": cfg}, last_path)
    dump_json({"best_val_qwk": best, "history": history},
              os.path.join(cfg["out_dir"], f"history_{args.tag}.json"))


if __name__ == "__main__":
    main()
