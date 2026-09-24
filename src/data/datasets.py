"""Unified CSV-driven datasets for EyePACS / DDR / APTOS (+ EyeQ quality labels).

Every split is described by a CSV with columns:
    image_path, grade, env, quality   (quality optional, may be NaN)

  grade    : int 0-4 (ICDR DR grade)
  env      : str in {eyepacs, ddr, aptos}  -> acquisition environment id
  quality  : str in {good, usable, reject} -> EyeQ human label (Fix 1 branch only)
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degradations import SyntheticDegradation
from .transforms import FundusPreprocess, eval_tensor_transform, train_tensor_transform

ENV2ID = {"eyepacs": 0, "ddr": 1, "aptos": 2, "deepdrid": 1, "idrid": 1}
ID2ENV = {v: k for k, v in ENV2ID.items()}
QUALITY2ID = {"reject": 0, "usable": 1, "good": 2}
QUALITY2SCORE = {"reject": 0.0, "usable": 0.5, "good": 1.0}
NUM_GRADES = 5


def _load_csv(csv_path: str, root: str = "") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"image_path", "grade", "env"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing column(s): {sorted(missing)}")
    if root:
        df["image_path"] = df["image_path"].apply(
            lambda p: p if os.path.isabs(p) else os.path.join(root, p))
    df["env"] = df["env"].str.lower().str.strip()
    unknown = set(df["env"]) - set(ENV2ID)
    if unknown:
        raise ValueError(f"unknown environment(s) {unknown}; expected {list(ENV2ID)}")
    return df


class FundusGradingDataset(Dataset):
    """Stage-2 dataset: image -> (grade, env_id, quality_label)."""

    def __init__(self, csv_path: str, root: str = "", size: int = 384,
                 train: bool = True, envs: Optional[list] = None):
        self.df = _load_csv(csv_path, root)
        if envs:
            self.df = self.df[self.df["env"].isin(envs)].reset_index(drop=True)
        self.pre = FundusPreprocess(size=size)
        self.tf = train_tensor_transform(size) if train else eval_tensor_transform(size)

    def __len__(self):
        return len(self.df)

    def class_counts(self) -> np.ndarray:
        c = np.zeros(NUM_GRADES, dtype=np.int64)
        for g, n in self.df["grade"].value_counts().items():
            c[int(g)] = n
        return c

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self.pre(Image.open(row["image_path"]))
        x = self.tf(img)
        q_lbl = -1
        if "quality" in row and isinstance(row.get("quality"), str):
            q_lbl = QUALITY2ID.get(row["quality"].lower().strip(), -1)
        return {
            "image": x,
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "env": torch.tensor(ENV2ID[row["env"]], dtype=torch.long),
            "quality_label": torch.tensor(q_lbl, dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }


class SelfSupervisedQualityDataset(Dataset):
    """Stage-1 dataset: two independently degraded views of the same image.

    Returns both views with their pseudo-quality targets so the encoder can be
    trained with (a) direct severity regression and (b) a pairwise ranking term
    that only needs the *relative* ordering -- no human quality annotation.
    """

    def __init__(self, csv_path: str, root: str = "", size: int = 384):
        self.df = _load_csv(csv_path, root)
        self.pre = FundusPreprocess(size=size)
        self.degrade = SyntheticDegradation()
        self.tf = train_tensor_transform(size, light=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        base = self.pre(Image.open(row["image_path"]))
        v1, q1 = self.degrade(base)
        v2, q2 = self.degrade(base)
        return {
            "view1": self.tf(v1),
            "view2": self.tf(v2),
            "q1": torch.tensor(q1, dtype=torch.float32),
            "q2": torch.tensor(q2, dtype=torch.float32),
        }


class QualityLabelDataset(Dataset):
    """Fix-1 calibration branch: real EyeQ Good/Usable/Reject labels."""

    def __init__(self, csv_path: str, root: str = "", size: int = 384, train: bool = True):
        df = _load_csv(csv_path, root)
        if "quality" not in df.columns:
            raise ValueError("calibration CSV must contain a 'quality' column")
        self.df = df.dropna(subset=["quality"]).reset_index(drop=True)
        self.pre = FundusPreprocess(size=size)
        self.tf = train_tensor_transform(size, light=True) if train else eval_tensor_transform(size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = self.pre(Image.open(row["image_path"]))
        return {
            "image": self.tf(img),
            "quality_label": torch.tensor(QUALITY2ID[row["quality"].lower().strip()],
                                          dtype=torch.long),
        }
