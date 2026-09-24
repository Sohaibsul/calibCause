"""Factory helpers shared by the training and evaluation entry points."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.datasets import NUM_GRADES, FundusGradingDataset
from ..models import (CalibratedQualityHead, QGDRPlus, QualityBranch,
                      QualityEncoder, SingleStreamStudent)


def build_quality_branch(cfg, device, mode: str | None = None) -> QualityBranch:
    mode = mode or cfg["model"].get("quality_mode", "calibrated")
    enc = QualityEncoder(cfg["model"]["quality_backbone"], pretrained=False,
                         embed_dim=cfg["model"]["embed_dim"])
    ck_path = cfg["model"].get("quality_encoder_ckpt")
    if ck_path and os.path.isfile(ck_path):
        enc.load_state_dict(torch.load(ck_path, map_location="cpu", weights_only=False)["model"])
    else:
        print(f"[warn] quality encoder checkpoint not found at {ck_path}; "
              f"using randomly initialised encoder (run Stage 1 first)")

    head = None
    if mode == "calibrated":
        head = CalibratedQualityHead(cfg["model"]["embed_dim"])
        hp = cfg["model"].get("quality_head_ckpt")
        if hp and os.path.isfile(hp):
            head.load_state_dict(torch.load(hp, map_location="cpu", weights_only=False)["head"])
        else:
            print(f"[warn] calibrated head checkpoint not found at {hp}")
    return QualityBranch(enc, head, mode=mode,
                         freeze_encoder=cfg["model"].get("freeze_quality", True)).to(device)


def build_teacher(cfg, device, quality_mode: str | None = None) -> QGDRPlus:
    qb = build_quality_branch(cfg, device, quality_mode)
    model = QGDRPlus(qb,
                     backbone_name=cfg["model"]["backbone"],
                     pretrained=cfg["model"]["pretrained"],
                     n_classes=NUM_GRADES,
                     n_envs=cfg["model"].get("n_envs", 3),
                     proj_dim=cfg["model"].get("proj_dim", 384),
                     use_causal=cfg["model"].get("use_causal", True),
                     use_env_disc=cfg["model"].get("use_env_disc", True),
                     gate_temperature=cfg["model"].get("gate_temperature", 1.0))
    return model.to(device)


def build_student(cfg, device) -> SingleStreamStudent:
    return SingleStreamStudent(cfg["student"]["backbone"], cfg["model"]["pretrained"],
                               n_classes=NUM_GRADES).to(device)


def build_loader(cfg, split: str, train: bool, envs=None, balanced: bool | None = None):
    ds = FundusGradingDataset(cfg["data"][f"{split}_csv"], cfg["data"].get("root", ""),
                              size=cfg["data"]["image_size"], train=train, envs=envs)
    balanced = cfg["train"].get("balanced_sampler", False) if balanced is None else balanced
    sampler, shuffle = None, train
    if train and balanced and len(ds):
        counts = ds.class_counts().astype(np.float64)
        w = 1.0 / np.clip(counts, 1, None)
        sw = w[ds.df["grade"].astype(int).to_numpy()]
        sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                        num_samples=len(ds), replacement=True)
        shuffle = False
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                      sampler=sampler, num_workers=cfg["train"]["num_workers"],
                      pin_memory=True, drop_last=train)


def class_weights_from(loader, n_classes: int = NUM_GRADES) -> torch.Tensor:
    counts = loader.dataset.class_counts().astype(np.float64)
    w = counts.sum() / (n_classes * np.clip(counts, 1, None))
    return torch.tensor(w, dtype=torch.float32)
