"""Synthetic fundus degradations for Stage-1 self-supervised quality pretraining.

The pipeline samples a scalar severity s ~ U(0, 1), applies a random subset of
degradation operators whose strength is scaled by s, and returns the degraded
image together with the *pseudo quality target* q = 1 - s.  No human quality
label is used anywhere here -- this is what makes Module 3 (label-free gating)
possible.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

OPS = ("blur", "noise", "jpeg", "illum", "haze", "defocus")


def _gaussian_blur(img: Image.Image, s: float) -> Image.Image:
    radius = 0.3 + 6.0 * s
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _defocus(img: Image.Image, s: float) -> Image.Image:
    # crude disc-kernel defocus: downsample then upsample
    w, h = img.size
    f = max(1.0, 1.0 + 7.0 * s)
    small = img.resize((max(8, int(w / f)), max(8, int(h / f))), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)


def _noise(img: Image.Image, s: float) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    sigma = 2.0 + 45.0 * s
    arr = arr + np.random.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg(img: Image.Image, s: float) -> Image.Image:
    q = int(round(95 - 88 * s))  # 95 -> 7
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=max(5, q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _illumination(img: Image.Image, s: float) -> Image.Image:
    """Uneven illumination: multiplicative radial/linear gradient (very common
    in real screening photographs -- overexposed arc on one side of the disc)."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = random.uniform(0.1, 0.9) * w, random.uniform(0.1, 0.9) * h
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = r / (r.max() + 1e-6)
    amp = 0.15 + 0.85 * s
    mask = 1.0 + amp * (0.5 - r)
    arr = arr * mask[..., None]
    out = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    return ImageEnhance.Contrast(out).enhance(1.0 - 0.45 * s)


def _haze(img: Image.Image, s: float) -> Image.Image:
    """Media opacity / cataract-like veiling glare: alpha-blend toward a bright
    low-frequency version of the image."""
    arr = np.asarray(img).astype(np.float32)
    veil = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=25))).astype(np.float32)
    veil = np.clip(veil * 1.15 + 25.0, 0, 255)
    a = 0.1 + 0.65 * s
    out = (1 - a) * arr + a * veil
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


_FN = {
    "blur": _gaussian_blur,
    "defocus": _defocus,
    "noise": _noise,
    "jpeg": _jpeg,
    "illum": _illumination,
    "haze": _haze,
}


class SyntheticDegradation:
    """Callable that returns (degraded_image, quality_target in [0, 1])."""

    def __init__(self, ops=OPS, min_ops: int = 1, max_ops: int = 3, p_clean: float = 0.15):
        self.ops = list(ops)
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.p_clean = p_clean  # keep some pristine samples so q=1 is anchored

    def __call__(self, img: Image.Image):
        if random.random() < self.p_clean:
            return img, 1.0
        s = random.random()
        k = random.randint(self.min_ops, self.max_ops)
        chosen = random.sample(self.ops, k=min(k, len(self.ops)))
        # per-op strength jitter around the global severity keeps the mapping
        # monotone but not degenerate
        for op in chosen:
            s_op = float(np.clip(s * random.uniform(0.75, 1.15), 0.0, 1.0))
            img = _FN[op](img, s_op)
        return img, float(1.0 - s)
