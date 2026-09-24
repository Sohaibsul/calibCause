"""Rebuild history_<tag>.json from the raw text log, for runs that were
interrupted (killed / system shutdown) before train_stage2.py reached its
final dump_json call.

Usage:
    python scripts/rebuild_history_from_log.py \
        --log runs/stage2/stage2_<tag>.log --tag eyepacs_only \
        --out-dir runs/stage2
"""
from __future__ import annotations

import argparse
import json
import os
import re

EP_PAT = re.compile(r"epoch (\d+): loss=([\d.]+) cls=([\d.]+) env=([\d.]+) "
                     r"ent=([\d.]+) acc=([\d.]+)")
VAL_PAT = re.compile(r"val acc=([\d.]+) qwk=([\d.]+) ece=([\d.]+)")


def parse(path: str):
    with open(path) as f:
        lines = f.readlines()
    hist = []
    i = 0
    while i < len(lines):
        m1 = EP_PAT.search(lines[i])
        if m1 and i + 1 < len(lines):
            m2 = VAL_PAT.search(lines[i + 1])
            if m2:
                ep, loss, cls, env, ent, acc = m1.groups()
                vacc, vqwk, vece = m2.groups()
                hist.append({
                    "epoch": int(ep),
                    "train": {"loss": float(loss), "cls": float(cls),
                              "env": float(env), "ent": float(ent), "acc": float(acc)},
                    "val": {"accuracy": float(vacc), "qwk": float(vqwk), "ece": float(vece)},
                })
                i += 2
                continue
        i += 1
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="runs/stage2")
    args = ap.parse_args()

    hist = parse(args.log)
    best = max((h["val"]["qwk"] for h in hist), default=-1.0)
    out_path = os.path.join(args.out_dir, f"history_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"best_val_qwk": best, "history": hist, "note": "rebuilt from log, run interrupted"}, f)
    print(f"parsed {len(hist)} epochs -> {out_path} (best qwk={best:.4f})")


if __name__ == "__main__":
    main()
