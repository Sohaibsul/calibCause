"""Shared matplotlib helpers for training curves and comparison tables.

Kept dependency-light (matplotlib + pandas only) and headless-safe (Agg
backend), since every plot in this project is produced on a remote GPU box
over SSH with no display.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "train": "#4C72B0",
    "val": "#DD8452",
    "teacher": "#4C72B0",
    "student": "#DD8452",
    "labelfree": "#55A868",
    "calibrated": "#4C72B0",
    "nocausal": "#C44E52",
    "eyepacs": "#4C72B0",
    "ddr": "#DD8452",
    "aptos": "#55A868",
}


def new_fig(nrows=1, ncols=1, figsize=None, sharey=False):
    figsize = figsize or (5.5 * ncols, 4.2 * nrows)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    return fig, ax


def save(fig, path: str, dpi: int = 160):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")
