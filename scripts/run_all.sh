#!/usr/bin/env bash
# Full experimental protocol. Edit the paths at the top, then: bash scripts/run_all.sh
set -euo pipefail

EYEPACS=/data/eyepacs
DDR=/data/DDR
APTOS=/data/aptos2019
EYEQ=/data/EyeQ

echo "=== 0. build CSV splits ==="
python scripts/prepare_data.py --eyepacs $EYEPACS --ddr $DDR --aptos $APTOS \
    --eyeq $EYEQ --out data/splits

echo "=== 1. Stage 1: self-supervised quality pretraining + Fix-1 calibration ==="
python -m src.engine.train_stage1 --config configs/stage1_ssl.yaml --calibrate

echo "=== 2. Stage 2a + Module 1: teacher, in-domain, calibrated gating ==="
python -m src.engine.train_stage2 --config configs/stage2_moe.yaml --tag teacher

echo "=== 3. Module 3 ablation: label-free gating ==="
python -m src.engine.train_stage2 --config configs/ablation_labelfree.yaml \
    --quality-mode labelfree --tag labelfree

echo "=== 4. Module 1 ablation: no causal block / no discriminator ==="
python -m src.engine.train_stage2 --config configs/ablation_nocausal.yaml \
    --no-causal --tag nocausal

echo "=== 5. leave-one-environment-out cross-environment generalisation ==="
# train on two environments, then evaluate on the held-out third (--split test
# is filtered to the held-out env at evaluation time via evaluate.py's
# per-environment breakdown -- the 'held-out' number is report["per_environment"][env])
python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
    --train-envs ddr aptos --val-envs ddr aptos --tag loeo_heldout_eyepacs
python -m src.engine.evaluate --config configs/stage2_moe.yaml \
    --ckpt runs/stage2/loeo_heldout_eyepacs_best.pt --split test \
    --tag loeo_heldout_eyepacs

python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
    --train-envs eyepacs aptos --val-envs eyepacs aptos --tag loeo_heldout_ddr
python -m src.engine.evaluate --config configs/stage2_moe.yaml \
    --ckpt runs/stage2/loeo_heldout_ddr_best.pt --split test --tag loeo_heldout_ddr

python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
    --train-envs eyepacs ddr --val-envs eyepacs ddr --tag loeo_heldout_aptos
python -m src.engine.evaluate --config configs/stage2_moe.yaml \
    --ckpt runs/stage2/loeo_heldout_aptos_best.pt --split test --tag loeo_heldout_aptos

echo "=== 6. Module 2: distil to single-stream student ==="
python -m src.engine.train_student --config configs/student_kd.yaml \
    --teacher-ckpt runs/stage2/teacher_best.pt

echo "=== 7. evaluation ==="
python -m src.engine.evaluate --config configs/stage2_moe.yaml \
    --ckpt runs/stage2/teacher_best.pt --split test --calibrate-temperature --tag teacher
python -m src.engine.evaluate --config configs/ablation_labelfree.yaml \
    --ckpt runs/ablation_labelfree/labelfree_best.pt --split test \
    --quality-mode labelfree --tag labelfree
python -m src.engine.evaluate --config configs/student_kd.yaml \
    --ckpt runs/student/student_best.pt --student --split test \
    --calibrate-temperature --tag student

echo "=== 8. interpretability ==="
python -m src.interpret.gradcam --config configs/stage2_moe.yaml \
    --ckpt runs/stage2/teacher_best.pt --n 32
python -m src.interpret.routing_audit --preds runs/stage2/preds_teacher_test.npz \
    --out runs/stage2/routing

echo "=== 9. calibrated uncertainty (reliability + risk-coverage) ==="
python -m src.interpret.uncertainty --preds runs/stage2/preds_teacher_test.npz \
    --val-preds runs/stage2/preds_teacher_val.npz --out runs/stage2/uncertainty
python -m src.interpret.uncertainty --preds runs/student/preds_student_test.npz \
    --val-preds runs/student/preds_student_val.npz --out runs/student/uncertainty

echo "=== 10. plots + base-paper comparison table ==="
python scripts/plot_curves.py --history runs/stage2/history_teacher.json \
    --out runs/stage2/curves_teacher.png
python scripts/plot_curves.py \
    --history runs/stage2/history_teacher.json:teacher \
    --history runs/ablation_labelfree/history_labelfree.json:labelfree \
    --history runs/ablation_nocausal/history_nocausal.json:nocausal \
    --metric qwk --out runs/comparison_qwk_curves.png
python scripts/plot_curves.py --stage1-log runs/stage1/stage1.log \
    --out runs/stage1/curves_ssl.png

python scripts/compare_to_qgdr.py \
    --teacher runs/stage2/report_teacher_test.json \
    --labelfree runs/ablation_labelfree/report_labelfree_test.json \
    --nocausal runs/ablation_nocausal/report_nocausal_test.json \
    --student runs/student/report_student_test.json \
    --loeo runs/stage2/report_loeo_heldout_eyepacs_test.json:held-out_EyePACS \
    --loeo runs/stage2/report_loeo_heldout_ddr_test.json:held-out_DDR \
    --loeo runs/stage2/report_loeo_heldout_aptos_test.json:held-out_APTOS \
    --out runs/comparison

echo "ALL DONE. reports in runs/*/report_*.json, plots in runs/*/*.png, table in runs/comparison/"
