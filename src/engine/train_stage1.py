"""Stage 1 -- self-supervised quality pretraining (+ optional Fix-1 calibration).

Run:
    python -m src.engine.train_stage1 --config configs/stage1_ssl.yaml
    python -m src.engine.train_stage1 --config configs/stage1_ssl.yaml --calibrate

Resume after an interruption (picks up from the epoch stored in the
checkpoint -- the script saves a checkpoint after every epoch, so at most one
epoch of work is ever lost):
    python -m src.engine.train_stage1 --config configs/stage1_ssl.yaml \
        --calibrate --resume
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets import QualityLabelDataset, SelfSupervisedQualityDataset
from ..losses import pairwise_ranking_loss, quality_regression_loss
from ..models import CalibratedQualityHead, QualityEncoder
from ..utils import (AverageMeter, device_from, dump_json, get_logger,
                     load_config, save_checkpoint, set_seed)


def train_ssl(cfg, device, logger, resume: bool = False):
    ds = SelfSupervisedQualityDataset(cfg["data"]["ssl_csv"], cfg["data"].get("root", ""),
                                      size=cfg["data"]["image_size"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    model = QualityEncoder(cfg["model"]["quality_backbone"],
                           pretrained=cfg["model"]["pretrained"],
                           embed_dim=cfg["model"]["embed_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")
    lam = cfg["train"].get("rank_weight", 0.5)

    start_ep = 0
    ckpt_path = os.path.join(cfg["out_dir"], "quality_encoder.pt")
    if resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_ep = ck["epoch"] + 1  # checkpoint stores the epoch just completed
        for _ in range(start_ep):
            sched.step()  # fast-forward the cosine schedule to match
        logger.info(f"resuming from epoch {start_ep + 1} "
                    f"(checkpoint was saved after epoch {ck['epoch'] + 1})")
    elif resume:
        logger.info("--resume given but no checkpoint found; starting from scratch")

    for ep in range(start_ep, cfg["train"]["epochs"]):
        model.train()
        m_reg, m_rank = AverageMeter(), AverageMeter()
        for b in tqdm(dl, desc=f"[stage1] epoch {ep + 1}/{cfg['train']['epochs']}"):
            v1, v2 = b["view1"].to(device), b["view2"].to(device)
            t1, t2 = b["q1"].to(device), b["q2"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                o1, o2 = model(v1), model(v2)
                l_reg = 0.5 * (quality_regression_loss(o1["q_logit"], t1) +
                               quality_regression_loss(o2["q_logit"], t2))
                l_rank = pairwise_ranking_loss(o1["q_raw"], o2["q_raw"], t1, t2)
                loss = l_reg + lam * l_rank
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            m_reg.update(l_reg.item(), v1.size(0))
            m_rank.update(l_rank.item(), v1.size(0))
        sched.step()
        logger.info(f"epoch {ep + 1}: reg={m_reg.avg:.4f} rank={m_rank.avg:.4f}")
        save_checkpoint({"model": model.state_dict(), "cfg": cfg, "epoch": ep},
                        os.path.join(cfg["out_dir"], "quality_encoder.pt"))
    return model


def train_calibration(cfg, device, logger, encoder=None):
    """Fix 1: freeze the SSL encoder, fit the 3-class head on real EyeQ labels."""
    if encoder is None:
        encoder = QualityEncoder(cfg["model"]["quality_backbone"], pretrained=False,
                                 embed_dim=cfg["model"]["embed_dim"]).to(device)
        ck = torch.load(os.path.join(cfg["out_dir"], "quality_encoder.pt"), map_location=device, weights_only=False)
        encoder.load_state_dict(ck["model"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    tr = QualityLabelDataset(cfg["data"]["eyeq_train_csv"], cfg["data"].get("root", ""),
                             cfg["data"]["image_size"], train=True)
    va = QualityLabelDataset(cfg["data"]["eyeq_val_csv"], cfg["data"].get("root", ""),
                             cfg["data"]["image_size"], train=False)
    dtr = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                     num_workers=cfg["train"]["num_workers"], pin_memory=True)
    dva = DataLoader(va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                     num_workers=cfg["train"]["num_workers"])

    head = CalibratedQualityHead(cfg["model"]["embed_dim"]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg["calibration"]["lr"], weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    best = -1.0
    for ep in range(cfg["calibration"]["epochs"]):
        head.train()
        for b in tqdm(dtr, desc=f"[fix1] epoch {ep + 1}"):
            x, y = b["image"].to(device), b["quality_label"].to(device)
            with torch.no_grad():
                emb = encoder(x)["emb"]
            loss = ce(head(emb)["quality_logits"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        preds, gts = [], []
        with torch.no_grad():
            for b in dva:
                emb = encoder(b["image"].to(device))["emb"]
                preds.append(head(emb)["quality_logits"].argmax(-1).cpu().numpy())
                gts.append(b["quality_label"].numpy())
        acc = float((np.concatenate(preds) == np.concatenate(gts)).mean())
        logger.info(f"[fix1] epoch {ep + 1}: val quality acc = {acc:.4f}")
        if acc > best:
            best = acc
            save_checkpoint({"head": head.state_dict(), "val_acc": acc},
                            os.path.join(cfg["out_dir"], "quality_head.pt"))
    dump_json({"best_quality_val_acc": best},
              os.path.join(cfg["out_dir"], "stage1_calibration.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--calibrate", action="store_true",
                    help="also fit the Fix-1 calibrated head on EyeQ labels")
    ap.add_argument("--only-calibrate", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="pick up SSL pretraining from the last saved epoch "
                         "in quality_encoder.pt instead of starting over")
    ap.add_argument("--opt", nargs="*", default=[], help="cfg overrides, e.g. train.epochs=1")
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = device_from(cfg)
    logger = get_logger("stage1", cfg["out_dir"])
    logger.info(f"device={device}")

    enc = None
    if not args.only_calibrate:
        enc = train_ssl(cfg, device, logger, resume=args.resume)
    if args.calibrate or args.only_calibrate:
        train_calibration(cfg, device, logger, enc)


if __name__ == "__main__":
    main()
