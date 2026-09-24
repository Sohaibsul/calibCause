"""Scan every split CSV for image_path entries that don't exist on disk.

Usage:
    # just report counts, don't change anything
    python scripts/check_missing_images.py --report-only

    # report AND write cleaned CSVs (backs up originals to <name>.csv.bak)
    python scripts/check_missing_images.py --fix
"""
from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

SPLITS = [
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/eyeq_train.csv",
    "data/splits/eyeq_val.csv",
    "data/splits/ssl_pool.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    grand_total, grand_missing = 0, 0
    for path in SPLITS:
        if not os.path.isfile(path):
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        exists = df["image_path"].apply(os.path.isfile)
        missing = df[~exists]
        grand_total += len(df)
        grand_missing += len(missing)

        by_env = missing["env"].value_counts().to_dict() if "env" in df.columns else {}
        print(f"{path}: total={len(df)} missing={len(missing)} by_env={by_env}")

        if len(missing) and args.fix:
            bak = path + ".bak"
            if not os.path.isfile(bak):
                shutil.copy(path, bak)
                print(f"  backed up original -> {bak}")
            cleaned = df[exists].reset_index(drop=True)
            cleaned.to_csv(path, index=False)
            print(f"  wrote cleaned {path} ({len(cleaned)} rows kept)")

    print(f"\nTOTAL: {grand_total} rows checked, {grand_missing} missing "
          f"({100 * grand_missing / max(grand_total, 1):.2f}%)")
    if grand_missing and not args.fix:
        print("Re-run with --fix to remove the missing rows and back up the originals.")


if __name__ == "__main__":
    main()
