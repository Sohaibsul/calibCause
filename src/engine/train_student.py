"""Module 2 -- distil the quality-gated MoE teacher into a single-stream student.

The student sees no quality branch, no gate and no experts; it must recover the
teacher's behaviour from soft logits alone.  The accuracy gap and the parameter
ratio together are the 'distillation trade-off' row of Figure 1's evaluation
block.

Run:
    python -m src.engine.train_student --config configs/student_kd.yaml \
        --teacher-ckpt runs/stage2/teacher_best.pt
"""
from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from ..losses import OrdinalAwareCE, distillation_loss
from ..utils import (AverageMeter, device_from, dump_json, get_logger,
                     load_config, save_checkpoint, set_seed)
from .builders import build_loader, build_student, build_teacher
from .evaluate import collect_predictions, evaluate_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--teacher-ckpt", required=True)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = device_from(cfg)
    logger = get_logger(f"kd_{args.tag}", cfg["out_dir"])

    teacher = build_teacher(cfg, device)
    tck = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(tck["model"], strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_student(cfg, device)
    n_t = sum(p.numel() for p in teacher.parameters())
    n_s = sum(p.numel() for p in student.parameters())
    logger.info(f"teacher params={n_t/1e6:.2f}M  student params={n_s/1e6:.2f}M  "
                f"ratio={n_s/n_t:.3f}")

    train_loader = build_loader(cfg, "train", train=True)
    val_loader = build_loader(cfg, "val", train=False)

    hard = OrdinalAwareCE(smoothing=cfg["train"].get("label_smoothing", 0.05),
                          ordinal_weight=cfg["train"].get("ordinal_weight", 0.5)).to(device)
    epochs = cfg["train"]["epochs"]
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    T = cfg["student"].get("temperature", 6.0)
    alpha = cfg["student"].get("alpha", 0.7)

    best, history = -1.0, []
    ckpt_path = os.path.join(cfg["out_dir"], f"{args.tag}_best.pt")
    for ep in range(epochs):
        student.train()
        m = {k: AverageMeter() for k in ("loss", "soft", "hard", "agree")}
        for b in tqdm(train_loader, desc=f"[kd] epoch {ep + 1}/{epochs}"):
            x, y = b["image"].to(device), b["grade"].to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                t_logits = teacher(x)["logits"].float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                s_logits = student(x)["logits"]
                d = distillation_loss(s_logits.float(), t_logits, y, T=T, alpha=alpha,
                                      hard_loss_fn=hard)
            scaler.scale(d["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            agree = (s_logits.argmax(-1) == t_logits.argmax(-1)).float().mean().item()
            for k, v in (("loss", d["loss"].item()), ("soft", float(d["soft"])),
                         ("hard", float(d["hard"])), ("agree", agree)):
                m[k].update(v, x.size(0))
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

    dump_json({"best_val_qwk": best, "teacher_params": n_t, "student_params": n_s,
               "param_ratio": n_s / n_t, "history": history},
              os.path.join(cfg["out_dir"], f"history_{args.tag}.json"))


if __name__ == "__main__":
    main()
