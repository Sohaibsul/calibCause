#!/bin/bash
set -e
cd ~/Grading_implementation
source .venv/bin/activate

# 1. Ablation: labelfree
python -m src.engine.train_stage2 --config configs/ablation_labelfree.yaml \
    --tag labelfree --opt train.epochs=15
python -m src.engine.evaluate --config configs/ablation_labelfree.yaml \
    --ckpt runs/ablation_labelfree/labelfree_best.pt --split test \
    --calibrate-temperature --tag labelfree

# 2. Ablation: nocausal
python -m src.engine.train_stage2 --config configs/ablation_nocausal.yaml \
    --tag nocausal --opt train.epochs=15
python -m src.engine.evaluate --config configs/ablation_nocausal.yaml \
    --ckpt runs/ablation_nocausal/nocausal_best.pt --split test \
    --calibrate-temperature --tag nocausal

# 3. Student distillation
python -m src.engine.train_student --config configs/student_kd.yaml \
    --teacher-ckpt runs/stage2/teacher_best.pt --opt train.epochs=15
python -m src.engine.evaluate --config configs/student_kd.yaml \
    --ckpt runs/student/student_best.pt --split test \
    --calibrate-temperature --tag student --student

echo "ALL DONE"
