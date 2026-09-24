"""Module 2 -- per-expert Grad-CAM.

Answers the question Figure 1 asks: *which pixels does each of the three
scale-specialised experts attend to?*  One CAM per expert per image, computed
against that expert's own logit rather than the fused logit, so the maps are
genuinely per-expert and not just three copies of the fused explanation.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.datasets import FundusGradingDataset
from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ..engine.builders import build_teacher
from ..utils import device_from, load_config

EXPERT_NAMES = ("fine_lesion", "vascular_lesion", "severe_dr")


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _colorize(cam: np.ndarray) -> np.ndarray:
    """Minimal jet-style colour map (avoids a hard matplotlib dependency here)."""
    c = np.clip(cam, 0, 1)
    r = np.clip(1.5 - np.abs(4 * c - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * c - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * c - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def per_expert_cams(model, x: torch.Tensor, target: int | None = None):
    """x: (1,3,H,W).  Returns dict with cams (3,H,W), gate weights and logits."""
    model.eval()
    model.zero_grad(set_to_none=True)
    out = model(x, grl_lambda=0.0, keep_maps=True)
    maps = model._feature_maps
    for m in maps:
        m.retain_grad()

    cls = int(out["logits"].argmax(-1).item()) if target is None else int(target)
    cams = []
    for i, fmap in enumerate(maps):
        score = out["expert_logits"][:, i, cls].sum()
        grad = torch.autograd.grad(score, fmap, retain_graph=True)[0]
        w = grad.mean(dim=(2, 3), keepdim=True)          # GAP over spatial dims
        cam = F.relu((w * fmap).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam.cpu().numpy())
    return {"cams": np.stack(cams), "target": cls,
            "gate_weights": out["gate_weights"][0].detach().cpu().numpy(),
            "quality": float(out["quality_score"][0]),
            "expert_pred": out["expert_logits"][0].argmax(-1).detach().cpu().numpy()}


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = _colorize(cam)
    blend = (1 - alpha) * img.astype(np.float32) + alpha * heat.astype(np.float32)
    return Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opt", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.opt)
    device = device_from(cfg)
    out_dir = args.out or os.path.join(cfg["out_dir"], "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    model = build_teacher(cfg, device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"], strict=False)

    ds = FundusGradingDataset(cfg["data"][f"{args.split}_csv"], cfg["data"].get("root", ""),
                              size=cfg["data"]["image_size"], train=False)
    idx = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)

    rows = []
    for k in idx:
        s = ds[int(k)]
        x = s["image"].unsqueeze(0).to(device)
        r = per_expert_cams(model, x)
        base = _denorm(s["image"])
        strip = [Image.fromarray(base)] + \
                [overlay(base, r["cams"][i]) for i in range(len(EXPERT_NAMES))]
        w, h = strip[0].size
        canvas = Image.new("RGB", (w * len(strip), h))
        for j, im in enumerate(strip):
            canvas.paste(im, (j * w, 0))
        name = f"{k:06d}_gt{int(s['grade'])}_pred{r['target']}.jpg"
        canvas.save(os.path.join(out_dir, name))
        rows.append({"index": int(k), "gt": int(s["grade"]), "pred": r["target"],
                     "quality": r["quality"],
                     **{f"gate_{n}": float(g) for n, g in zip(EXPERT_NAMES, r["gate_weights"])}})

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "gradcam_index.csv"), index=False)
    print(f"wrote {len(rows)} panels (original | fine | vascular | severe) -> {out_dir}")


if __name__ == "__main__":
    main()
