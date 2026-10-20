"""Module 2 -- distil the quality-gated MoE teacher into a single-stream student.

Resume-safe and collapse-safe version of train_student.py. Differences:
  * teacher: reports missing/unexpected keys and measures its validation QWK before training (exit 3 if broken)
  * linear lr warmup over the first iterations of a fresh run + gradient clipping (stabilises the student)
  * early collapse guard: if the running teacher-agreement is too low at iteration N of epoch 1, exit 5
  * <tag>_last.pt after every epoch (atomic) and --resume; history_<tag>.json after every epoch
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from tqdm import tqdm

from ..losses import OrdinalAwareCE, distillation_loss
from ..utils import (AverageMeter, device_from, dump_json, get_logger,
                     load_config, save_checkpoint, set_seed)
from .builders import build_loader, build_student, build_teacher
from .evaluate import collect_predictions, evaluate_predictions


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-teacher-qwk", type=float, default=0.6)
    ap.add_argument("--skip-teacher-check", action="store_true")
    ap.add_argument("--warmup-iters", type=int, default=300)
    ap.add_argument("--collapse-check-iter", type=int, default=1200)
    ap.add_argument("--collapse-agree", type=float, default=0.77)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = device_from(cfg)
    logger = get_logger(f"kd_{args.tag}", cfg["out_dir"])
    logger.info(f"settings: lr={cfg['train']['lr']} epochs={cfg['train']['epochs']} warmup_iters={args.warmup_iters} "
                f"clip={cfg['train'].get('clip_grad', 5.0)} quality_mode={cfg['model'].get('quality_mode')}")

    # ---- teacher: load, report key mismatches, freeze
    teacher = build_teacher(cfg, device)
    tck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    res = teacher.load_state_dict(tck["model"], strict=False)
    n_keys = len(teacher.state_dict())
    logger.info(f"teacher load: missing={len(res.missing_keys)}/{n_keys} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        logger.info(f"  missing (first 8): {res.missing_keys[:8]}")
    if res.unexpected_keys:
        logger.info(f"  unexpected (first 8): {res.unexpected_keys[:8]}")
    if len(res.missing_keys) > 0.1 * n_keys:
        logger.error("teacher checkpoint does not match the model config -- aborting")
        sys.exit(3)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_student(cfg, device)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    logger.info(f"teacher params={n_t/1e6:.2f}M  student params={n_s/1e6:.2f}M  ratio={n_s/n_t:.3f}")

    train_loader = build_loader(cfg, "train", train=True)
    val_loader = build_loader(cfg, "val", train=False)

    hard = OrdinalAwareCE(smoothing=cfg["train"].get("label_smoothing", 0.05),
                          ordinal_weight=cfg["train"].get("ordinal_weight", 0.5)).to(device)
    epochs = cfg["train"]["epochs"]
    base_lr = cfg["train"]["lr"]
    clip = cfg["train"].get("clip_grad", 5.0)
    bb_mult = cfg["train"].get("lr_backbone_mult", 0.1)
    opt = torch.optim.AdamW([{"params": list(student.body.parameters()), "lr": base_lr * bb_mult, "base": base_lr * bb_mult},
                             {"params": list(student.head.parameters()), "lr": base_lr, "base": base_lr}],
                            weight_decay=cfg["train"]["weight_decay"])
    logger.info(f"optimizer: head lr={base_lr:.2e}  backbone lr={base_lr * bb_mult:.2e} (lr_backbone_mult={bb_mult})")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    T = cfg["student"].get("temperature", 6.0)
    alpha = cfg["student"].get("alpha", 0.7)

    best, history, start_epoch = -1.0, [], 0
    ckpt_path = os.path.join(cfg["out_dir"], f"{args.tag}_best.pt")
    last_path = os.path.join(cfg["out_dir"], f"{args.tag}_last.pt")
    hist_path = os.path.join(cfg["out_dir"], f"history_{args.tag}.json")

    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        logger.info(f"resumed from {last_path}: continuing at epoch {start_epoch + 1}/{epochs}, best qwk so far {best:.4f}")

    # ---- teacher sanity check (fresh start only)
    if start_epoch == 0 and not args.skip_teacher_check:
        try:
            tp = collect_predictions(teacher, val_loader, device, is_student=False)
            trep = evaluate_predictions(tp, per_env=True)
            tq = trep["overall"]["qwk"]
            logger.info(f"teacher sanity check: val acc={trep['overall']['accuracy']:.4f} qwk={tq:.4f}")
            if tq < args.min_teacher_qwk:
                logger.error(f"teacher val QWK {tq:.4f} < {args.min_teacher_qwk} -- teacher not loaded correctly, aborting")
                sys.exit(3)
        except Exception as e:  # SystemExit is not caught here
            logger.warning(f"teacher sanity check skipped: {e!r}")

    for ep in range(start_epoch, epochs):
        student.train()
        warm = (ep == 0 and start_epoch == 0)       # warmup only on a fresh run
        m = {k: AverageMeter() for k in ("loss", "soft", "hard", "agree")}
        for it, b in enumerate(tqdm(train_loader, desc=f"[kd] epoch {ep + 1}/{epochs}")):
            if warm and it < args.warmup_iters:
                for g in opt.param_groups:
                    g["lr"] = g["base"] * (it + 1) / args.warmup_iters
            elif warm and it == args.warmup_iters:
                for g in opt.param_groups:
                    g["lr"] = g["base"]
            x, y = b["image"].to(device), b["grade"].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                t_logits = teacher(x)["logits"].float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                s_logits = student(x)["logits"]
                d = distillation_loss(s_logits.float(), t_logits, y, T=T, alpha=alpha,
                                      hard_loss_fn=hard)
            scaler.scale(d["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            agree = (s_logits.argmax(-1) == t_logits.argmax(-1)).float().mean().item()
            for k, v in (("loss", d["loss"].item()), ("soft", float(d["soft"])),
                         ("hard", float(d["hard"])), ("agree", agree)):
                m[k].update(v, x.size(0))
            if warm and (it + 1) % args.log_every == 0:
                logger.info(f"  it {it + 1}: loss={m['loss'].avg:.4f} agree={m['agree'].avg:.4f} "
                            f"lr={opt.param_groups[0]['lr']:.2e}")
            if warm and (it + 1) == args.collapse_check_iter and m["agree"].avg < args.collapse_agree:
                logger.error(f"collapse guard: running teacher-agreement {m['agree'].avg:.4f} < {args.collapse_agree} "
                             f"at iteration {it + 1} -- student is not learning, exit 5")
                sys.exit(5)
        sched.step()

        pred = collect_predictions(student, val_loader, device, is_student=True)
        rep = evaluate_predictions(pred, per_env=True)
        logger.info(f"epoch {ep + 1}: loss={m['loss'].avg:.4f} "
                    f"teacher-agreement={m['agree'].avg:.4f} "
                    f"val acc={rep['overall']['accuracy']:.4f} qwk={rep['overall']['qwk']:.4f}")
        history.append({"epoch": ep + 1, "train": {k: v.avg for k, v in m.items()},
                        "val": rep["overall"]})
        if rep["overall"]["qwk"] > best:
            best = rep["overall"]["qwk"]
            save_checkpoint({"model": student.state_dict(), "cfg": cfg, "epoch": ep,
                             "val": rep, "params": n_s}, ckpt_path)
        _atomic_save({"student": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "scaler": scaler.state_dict(), "epoch": ep, "best": best, "history": history}, last_path)
        dump_json({"best_val_qwk": best, "teacher_params": n_t, "student_params": n_s,
                   "param_ratio": n_s / n_t, "history": history}, hist_path)


if __name__ == "__main__":
    main()
