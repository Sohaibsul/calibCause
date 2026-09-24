"""Build the unified CSVs every training script consumes.

Output schema (one row per image):
    image_path, grade, env, quality

Expected raw layouts
--------------------
EyePACS (Kaggle 2015 "Diabetic Retinopathy Detection"):
    <root>/trainLabels.csv          columns: image, level
    <root>/train/<image>.jpeg
DDR (nkicsl/DDR-dataset, DR_grading):
    <root>/DR_grading/{train,valid,test}.txt   lines: "<name>.jpg <grade>"
    <root>/DR_grading/{train,valid,test}/<name>.jpg
APTOS 2019:
    <root>/train.csv                columns: id_code, diagnosis
    <root>/train_images/<id_code>.png
EyeQ (quality labels for EyePACS images):
    <root>/Label_EyeQ_train.csv     columns: image, quality   (0=Good,1=Usable,2=Reject)

Example
-------
python scripts/prepare_data.py \
    --eyepacs /data/eyepacs --ddr /data/DDR --aptos /data/aptos2019 \
    --eyeq /data/EyeQ --out data/splits
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

EYEQ_ID2NAME = {0: "good", 1: "usable", 2: "reject"}


def _exists(df: pd.DataFrame, check: bool) -> pd.DataFrame:
    if not check:
        return df
    keep = df["image_path"].apply(os.path.isfile)
    if (~keep).any():
        print(f"  [warn] dropping {int((~keep).sum())} rows with missing files")
    return df[keep].reset_index(drop=True)


def load_eyepacs(root: str, check: bool) -> pd.DataFrame:
    lab = pd.read_csv(os.path.join(root, "trainLabels.csv"))
    img_dir = os.path.join(root, "train")
    df = pd.DataFrame({
        "image_path": lab["image"].apply(lambda n: os.path.join(img_dir, f"{n}.jpeg")),
        "grade": lab["level"].astype(int),
        "env": "eyepacs",
        "image_id": lab["image"].astype(str),
    })
    return _exists(df, check)


def load_ddr(root: str, check: bool, keep_ungradable: bool = False) -> pd.DataFrame:
    base = os.path.join(root, "DR_grading")
    frames = []
    for split in ("train", "valid", "test"):
        txt = os.path.join(base, f"{split}.txt")
        if not os.path.isfile(txt):
            print(f"  [warn] {txt} not found, skipping")
            continue
        rec = pd.read_csv(txt, sep=r"\s+", header=None, names=["name", "grade"])
        rec["image_path"] = rec["name"].apply(lambda n: os.path.join(base, split, n))
        rec["orig_split"] = split
        frames.append(rec)
    df = pd.concat(frames, ignore_index=True)
    df["grade"] = df["grade"].astype(int)
    if not keep_ungradable:
        n = int((df["grade"] == 5).sum())
        if n:
            print(f"  [info] dropping {n} DDR images labelled 5 (ungradable)")
        df = df[df["grade"] != 5].reset_index(drop=True)
    df["env"] = "ddr"
    df["image_id"] = df["name"].str.replace(r"\.\w+$", "", regex=True)
    return _exists(df[["image_path", "grade", "env", "image_id", "orig_split"]], check)


def load_aptos(root: str, check: bool) -> pd.DataFrame:
    lab = pd.read_csv(os.path.join(root, "train.csv"))
    img_dir = os.path.join(root, "train_images")
    df = pd.DataFrame({
        "image_path": lab["id_code"].apply(lambda n: os.path.join(img_dir, f"{n}.png")),
        "grade": lab["diagnosis"].astype(int),
        "env": "aptos",
        "image_id": lab["id_code"].astype(str),
    })
    return _exists(df, check)


def attach_eyeq(df: pd.DataFrame, eyeq_root: str) -> pd.DataFrame:
    """EyeQ annotates a subset of EyePACS; merge on image id."""
    frames = []
    for f in ("Label_EyeQ_train.csv", "Label_EyeQ_test.csv"):
        p = os.path.join(eyeq_root, f)
        if os.path.isfile(p):
            frames.append(pd.read_csv(p))
    if not frames:
        print("  [warn] no EyeQ label files found; 'quality' column left empty")
        df["quality"] = np.nan
        return df
    eq = pd.concat(frames, ignore_index=True)
    col = "quality" if "quality" in eq.columns else eq.columns[-1]
    eq["image_id"] = eq["image"].astype(str).str.replace(r"\.\w+$", "", regex=True)
    eq["quality"] = eq[col].astype(int).map(EYEQ_ID2NAME)
    df = df.merge(eq[["image_id", "quality"]], on="image_id", how="left")
    print(f"  [info] attached EyeQ quality labels to {int(df['quality'].notna().sum())} images")
    return df


def stratified_split(df: pd.DataFrame, ratios=(0.7, 0.1, 0.2), seed: int = 42):
    """Stratify by (env, grade) so every split keeps the same class mix per site."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby(["env", "grade"]):
        idx = rng.permutation(len(grp))
        n = len(grp)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        g = grp.iloc[idx]
        parts["train"].append(g.iloc[:n_tr])
        parts["val"].append(g.iloc[n_tr:n_tr + n_va])
        parts["test"].append(g.iloc[n_tr + n_va:])
    return {k: pd.concat(v, ignore_index=True).sample(frac=1, random_state=seed)
            .reset_index(drop=True) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eyepacs", default=None)
    ap.add_argument("--ddr", default=None)
    ap.add_argument("--aptos", default=None)
    ap.add_argument("--eyeq", default=None)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ssl-per-env", type=int, default=8000,
                    help="images sampled per environment for the Stage-1 SSL pool")
    ap.add_argument("--no-check-files", action="store_true")
    ap.add_argument("--keep-ungradable", action="store_true")
    args = ap.parse_args()

    check = not args.no_check_files
    os.makedirs(args.out, exist_ok=True)
    frames = []
    if args.eyepacs:
        print("loading EyePACS...")
        frames.append(load_eyepacs(args.eyepacs, check))
    if args.ddr:
        print("loading DDR...")
        frames.append(load_ddr(args.ddr, check, args.keep_ungradable))
    if args.aptos:
        print("loading APTOS...")
        frames.append(load_aptos(args.aptos, check))
    if not frames:
        raise SystemExit("give at least one of --eyepacs / --ddr / --aptos")

    df = pd.concat(frames, ignore_index=True)
    df = attach_eyeq(df, args.eyeq) if args.eyeq else df.assign(quality=np.nan)

    print("\nper-environment grade distribution:")
    print(pd.crosstab(df["env"], df["grade"]))

    splits = stratified_split(df, seed=args.seed)
    cols = ["image_path", "grade", "env", "quality"]
    for name, part in splits.items():
        part[cols].to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(f"{name}: {len(part)}")

    # Stage-1 SSL pool: images only, no grade or quality supervision used
    # NOTE: groupby(...).apply() drops the grouping column in pandas >= 3.0,
    # so the per-environment sample is built explicitly instead.
    ssl = pd.concat(
        [g.sample(min(len(g), args.ssl_per_env), random_state=args.seed)
         for _, g in splits["train"].groupby("env")], ignore_index=True)
    ssl[cols].to_csv(os.path.join(args.out, "ssl_pool.csv"), index=False)
    print(f"ssl_pool: {len(ssl)}")

    # Fix-1 calibration split (EyeQ-labelled subset only)
    q = df.dropna(subset=["quality"])
    if len(q):
        qs = stratified_split(q.assign(grade=q["grade"]), ratios=(0.8, 0.2, 0.0),
                              seed=args.seed)
        qs["train"][cols].to_csv(os.path.join(args.out, "eyeq_train.csv"), index=False)
        qs["val"][cols].to_csv(os.path.join(args.out, "eyeq_val.csv"), index=False)
        print(f"eyeq_train: {len(qs['train'])}  eyeq_val: {len(qs['val'])}")
    print(f"\nwrote CSVs to {args.out}")


if __name__ == "__main__":
    main()
