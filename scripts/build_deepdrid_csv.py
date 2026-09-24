"""Build a DeepDRiD external test CSV in the project schema
(image_path, grade, env, quality) from the regular-fundus-validation set.

DeepDRiD stores the DR grade per eye: for a *_l* image the grade is in
left_eye_DR_Level, for a *_r* image in right_eye_DR_Level. Paths in the
source CSV are Windows-style and relative; we rewrite them to absolute
POSIX paths under --root.

Usage:
    python scripts/build_deepdrid_csv.py \
        --root ~/DeepDRiD/regular_fundus_images/regular-fundus-validation \
        --labels ~/DeepDRiD/regular_fundus_images/regular-fundus-validation/regular-fundus-validation.csv \
        --out data/splits/deepdrid.csv
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder that contains the numbered patient subfolders")
    ap.add_argument("--labels", required=True, help="regular-fundus-validation.csv")
    ap.add_argument("--out", default="data/splits/deepdrid.csv")
    args = ap.parse_args()

    root = os.path.abspath(os.path.expanduser(args.root))
    df = pd.read_csv(args.labels)

    rows = []
    missing = 0
    for _, r in df.iterrows():
        image_id = str(r["image_id"])
        # choose the grade for this eye
        if image_id.split("_")[-1].startswith("l"):
            grade = r.get("left_eye_DR_Level")
        else:
            grade = r.get("right_eye_DR_Level")
        if pd.isna(grade):
            continue
        grade = int(grade)
        if grade not in (0, 1, 2, 3, 4):   # keep only the five ordinal grades
            continue

        # rebuild the path: source is like \regular-fundus-validation\265\265_l1.jpg
        # we only need the patient-folder + filename under --root
        raw = str(r["image_path"]).replace("\\", "/").strip("/")
        parts = raw.split("/")
        # last two components are <patient_folder>/<filename>
        rel = os.path.join(parts[-2], parts[-1])
        abspath = os.path.join(root, rel)
        if not os.path.isfile(abspath):
            missing += 1
            continue

        # quality: DeepDRiD "Overall quality" 0/1 (1 = gradable). Map loosely to
        # a quality string; the pipeline only needs the column to exist.
        q = r.get("Overall quality")
        quality = "" if pd.isna(q) else ("good" if int(q) == 1 else "usable")

        rows.append({"image_path": abspath, "grade": grade,
                     "env": "deepdrid", "quality": quality})

    out = pd.DataFrame(rows, columns=["image_path", "grade", "env", "quality"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} images  (missing on disk: {missing})")
    print("grade distribution:", out["grade"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
