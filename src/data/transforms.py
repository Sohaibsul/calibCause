"""Fundus preprocessing + augmentation."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_black_borders(img: Image.Image, tol: int = 7) -> Image.Image:
    """Remove the black frame around the retinal circle (standard for EyePACS)."""
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows, cols = np.where(mask)
    arr = arr[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    if arr.size == 0:
        return img
    return Image.fromarray(arr)


def circular_mask(img: Image.Image) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2), 255, -1)
    arr = cv2.bitwise_and(arr, arr, mask=mask)
    return Image.fromarray(arr)


class FundusPreprocess:
    """Deterministic geometric preprocessing applied BEFORE any degradation.

    NOTE: no CLAHE / Ben-Graham colour normalisation is applied by default --
    those operators partially undo the very acquisition artefacts the quality
    branch is supposed to see.  Enable only for the ablation.
    """

    def __init__(self, size: int = 384, do_crop: bool = True, do_mask: bool = False):
        self.size = size
        self.do_crop = do_crop
        self.do_mask = do_mask

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if self.do_crop:
            img = crop_black_borders(img)
        img = img.resize((self.size, self.size), Image.BILINEAR)
        if self.do_mask:
            img = circular_mask(img)
        return img


def train_tensor_transform(size: int = 384, light: bool = False):
    """Photometric/geometric augmentation -> tensor.

    `light=True` is used for the quality branch: strong colour jitter would
    fight the synthetic degradation targets.
    """
    aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
           T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05))]
    if not light:
        aug.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02))
    aug += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(aug)


def eval_tensor_transform(size: int = 384):
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
