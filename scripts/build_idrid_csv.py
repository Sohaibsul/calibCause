"""Build an IDRiD external test CSV in the project schema."""
from __future__ import annotations
import argparse, os
import pandas as pd


def load_split(base, images_subdir, labels_csv):
    img_dir = os.path.join(base, "1. Original Images", images_subdir)
    lab = os.path.join(base, "2. Groundtruths", labels_csv)
    df = pd.read_csv(lab)
    df.columns = [c.strip() for c in df.columns]
    rows, missing = [], 0
    for _, r in df.iterrows():
        name = str(r["Image name"]).strip()
        if not name or name.lower() == "nan":
            continue
        grade = r["Retinopathy grade"]
        if pd.isna(grade):
            continue
        grade = int(grade)
        if grade not in (0, 1, 2, 3, 4):
            continue
        path = os.path.join(img_dir, name + ".jpg")
        if not os.path.isfile(path):
            missing += 1
            continue
        rows.append({"image_path": path, "grade": grade, "env": "idrid", "quality": ""})
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", default="data/splits/idrid.csv")
    ap.add_argument("--include-train", action="store_true")
    args = ap.parse_args()
    base = os.path.expanduser(args.base)
    rows, missing = load_split(base, "b. Testing Set",
                               "b. IDRiD_Disease Grading_Testing Labels.csv")
    if args.include_train:
        r2, m2 = load_split(base, "a. Training Set",
                            "a. IDRiD_Disease Grading_Training Labels.csv")
        rows += r2
        missing += m2
    out = pd.DataFrame(rows, columns=["image_path", "grade", "env", "quality"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} images  (missing on disk: {missing})")
    print("grade distribution:", out["grade"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
