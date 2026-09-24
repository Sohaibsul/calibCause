#!/bin/bash
cd ~/Grading_implementation
source .venv/bin/activate
CFG=configs/stage2_moe.yaml

# per-source
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/ddr_only_best.pt --split test --calibrate-temperature --tag ddr_only --envs ddr
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/aptos_only_best.pt --split test --calibrate-temperature --tag aptos_only --envs aptos
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/eyepacs_only_best.pt --split test --calibrate-temperature --tag eyepacs_only --envs eyepacs

# combined teacher
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/teacher_best.pt --split test --calibrate-temperature --tag teacher

# LOEO
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/loeo_heldout_eyepacs_best.pt --split test --calibrate-temperature --tag loeo_heldout_eyepacs --envs eyepacs
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/loeo_heldout_ddr_best.pt --split test --calibrate-temperature --tag loeo_heldout_ddr --envs ddr
python -m src.engine.evaluate --config $CFG --ckpt runs/stage2/loeo_heldout_aptos_best.pt --split test --calibrate-temperature --tag loeo_heldout_aptos --envs aptos

# ablations
python -m src.engine.evaluate --config configs/ablation_labelfree.yaml --ckpt runs/ablation_labelfree/labelfree_best.pt --split test --calibrate-temperature --tag labelfree
python -m src.engine.evaluate --config configs/ablation_nocausal.yaml --ckpt runs/ablation_nocausal/nocausal_best.pt --split test --calibrate-temperature --tag nocausal

# student
python -m src.engine.evaluate --config configs/student_kd.yaml --ckpt runs/student/student_best.pt --split test --calibrate-temperature --tag student --student

echo "ALL EVAL DONE"
