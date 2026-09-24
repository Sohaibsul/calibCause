"""End-to-end smoke test on synthetic images -- no real data, no GPU needed.

Generates a tiny fake corpus, then runs Stage 1 -> Fix 1 -> Stage 2 -> KD ->
evaluation -> Grad-CAM -> routing audit with 1 epoch each.  Use this to verify
the environment before launching a real run.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = ["eyepacs", "ddr", "aptos"]


def make_fake_corpus(base: str, n_per_env: int = 24, size: int = 128):
    img_dir = os.path.join(base, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for e_i, env in enumerate(ENVS):
        for i in range(n_per_env):
            grade = i % 5
            # environment-specific colour cast + grade-specific bright specks,
            # so the adversary and the experts both have something to latch onto
            arr = rng.integers(40, 90, (size, size, 3), dtype=np.uint8)
            arr[..., e_i] = np.clip(arr[..., e_i].astype(int) + 60, 0, 255)
            for _ in range(grade * 12):
                y, x = rng.integers(0, size, 2)
                arr[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = 240
            p = os.path.join(img_dir, f"{env}_{i:04d}.png")
            Image.fromarray(arr).save(p)
            rows.append({"image_path": p, "grade": grade, "env": env,
                         "quality": ["good", "usable", "reject"][i % 3]})
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    out = os.path.join(base, "splits")
    os.makedirs(out, exist_ok=True)
    n = len(df)
    df.iloc[: int(0.6 * n)].to_csv(os.path.join(out, "train.csv"), index=False)
    df.iloc[int(0.6 * n): int(0.8 * n)].to_csv(os.path.join(out, "val.csv"), index=False)
    df.iloc[int(0.8 * n):].to_csv(os.path.join(out, "test.csv"), index=False)
    df.to_csv(os.path.join(out, "ssl_pool.csv"), index=False)
    df.iloc[: int(0.8 * n)].to_csv(os.path.join(out, "eyeq_train.csv"), index=False)
    df.iloc[int(0.8 * n):].to_csv(os.path.join(out, "eyeq_val.csv"), index=False)
    return out


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    tmp = tempfile.mkdtemp(prefix="qgdr_smoke_")
    splits = make_fake_corpus(tmp)
    runs = os.path.join(tmp, "runs")
    py = sys.executable
    common = [
        f"data.root=", f"data.train_csv={splits}/train.csv", f"data.val_csv={splits}/val.csv",
        f"data.test_csv={splits}/test.csv", f"data.ssl_csv={splits}/ssl_pool.csv",
        f"data.eyeq_train_csv={splits}/eyeq_train.csv", f"data.eyeq_val_csv={splits}/eyeq_val.csv",
        "data.image_size=128", "train.batch_size=4", "train.num_workers=0",
        "train.epochs=1", "train.amp=false", "model.pretrained=false",
        "model.backbone=convnext_tiny", "model.quality_backbone=convnext_tiny",
        "device=cpu",
    ]
    st1 = common + [f"out_dir={runs}/stage1", "calibration.epochs=1"]
    st2 = common + [f"out_dir={runs}/stage2",
                    f"model.quality_encoder_ckpt={runs}/stage1/quality_encoder.pt",
                    f"model.quality_head_ckpt={runs}/stage1/quality_head.pt"]
    kd = common + [f"out_dir={runs}/student",
                   f"model.quality_encoder_ckpt={runs}/stage1/quality_encoder.pt",
                   f"model.quality_head_ckpt={runs}/stage1/quality_head.pt"]

    run([py, "-m", "src.engine.train_stage1", "--config", "configs/stage1_ssl.yaml",
         "--calibrate", "--opt", *st1])
    run([py, "-m", "src.engine.train_stage2", "--config", "configs/stage2_moe.yaml",
         "--tag", "teacher", "--opt", *st2])
    run([py, "-m", "src.engine.train_stage2", "--config", "configs/stage2_moe.yaml",
         "--quality-mode", "labelfree", "--tag", "labelfree", "--opt", *st2])
    run([py, "-m", "src.engine.train_stage2", "--config", "configs/stage2_moe.yaml",
         "--train-envs", "eyepacs", "ddr", "--val-envs", "aptos", "--tag", "loeo", "--opt", *st2])
    run([py, "-m", "src.engine.evaluate", "--config", "configs/stage2_moe.yaml",
         "--ckpt", f"{runs}/stage2/teacher_best.pt", "--split", "test",
         "--calibrate-temperature", "--tag", "teacher", "--opt", *st2])
    run([py, "-m", "src.engine.train_student", "--config", "configs/student_kd.yaml",
         "--teacher-ckpt", f"{runs}/stage2/teacher_best.pt", "--opt", *kd])
    run([py, "-m", "src.engine.evaluate", "--config", "configs/student_kd.yaml",
         "--ckpt", f"{runs}/student/student_best.pt", "--student", "--split", "test",
         "--tag", "student", "--opt", *kd])
    run([py, "-m", "src.interpret.gradcam", "--config", "configs/stage2_moe.yaml",
         "--ckpt", f"{runs}/stage2/teacher_best.pt", "--n", "3", "--opt", *st2])
    run([py, "-m", "src.interpret.routing_audit",
         "--preds", f"{runs}/stage2/preds_teacher_test.npz",
         "--out", f"{runs}/stage2/routing"])
    run([py, "-m", "src.interpret.uncertainty",
         "--preds", f"{runs}/stage2/preds_teacher_test.npz",
         "--val-preds", f"{runs}/stage2/preds_teacher_val.npz",
         "--out", f"{runs}/stage2/uncertainty"])

    print(f"\nALL STAGES PASSED. artefacts in {runs}")


if __name__ == "__main__":
    main()
