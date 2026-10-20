"""Config loading, seeding, checkpointing, logging helpers."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("_base_", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        merged = load_config(base_path)
        merged = deep_update(merged, cfg)
        cfg = merged
    for ov in overrides or []:          # e.g.  train.epochs=5
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = yaml.safe_load(val)
        except yaml.YAMLError:
            node[parts[-1]] = val
    return cfg


def deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_update(a[k], v)
        else:
            a[k] = v
    return a


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_checkpoint(state: dict, path: str):
    import time as _time
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # atomic write + retry (2026-10-06: truncated _best.pt after "unexpected pos ... vs ...")
    tmp = path + ".tmp"
    last_err = None
    for attempt in range(3):
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)
            return
        except Exception as e:
            last_err = e
            print(f"[save_checkpoint] attempt {attempt + 1}/3 failed for {path}: {e}", flush=True)
            if os.path.exists(tmp):
                os.remove(tmp)
            _time.sleep(5)
    raise last_err


def dump_json(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def device_from(cfg: dict) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
