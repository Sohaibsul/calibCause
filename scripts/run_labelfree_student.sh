#!/usr/bin/env bash
# Distil a single-stream student from the label-free teacher (Module 3 variant).
# Output goes to runs/student_labelfree (tag student_lf); the existing Full-model student is not touched.
set -u
cd "$HOME/Grading_implementation" || exit 1
PY="$PWD/.venv/bin/python"
TEACHER=runs/ablation_labelfree/labelfree_best.pt
OUT=runs/student_labelfree
TAG=student_lf
EPOCHS=15
MIN_GPU_MB=3500
MIN_RAM_MB=6144
mkdir -p logs backups

# Only one launcher at a time.
exec 6>/tmp/calibcause_lfstudent_launcher.lock
flock -n 6 || { echo "[lfstudent] already running, exiting"; exit 0; }
# Share the queue with the other CalibCause jobs.
exec 9>/tmp/calibcause_queue.lock
echo "[lfstudent $(date '+%m-%d %H:%M')] waiting for queue lock"
flock -w 54000 9 || { echo "[lfstudent] queue lock timeout"; exit 1; }

[ -f "$TEACHER" ] || { echo "[lfstudent] teacher checkpoint missing: $TEACHER"; exit 1; }

# Resource gate: GPU free and host RAM available on 3 consecutive checks.
ok=0
while [ "$ok" -lt 3 ]; do
  gpu_free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}' | head -1)
  ram_av=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
  if [ "${gpu_free:-0}" -ge "$MIN_GPU_MB" ] && [ "${ram_av:-0}" -ge "$MIN_RAM_MB" ]; then ok=$((ok+1)); else ok=0; fi
  [ "$ok" -lt 3 ] && sleep 30
done
echo "[lfstudent $(date '+%m-%d %H:%M')] resources OK (gpu_free=${gpu_free}MB ram_avail=${ram_av}MB), starting"

# Protect the existing Full-model student and any earlier run in the output dir.
[ -f runs/student/student_best.pt ] && cp -n runs/student/student_best.pt backups/student_full_best_backup.pt
if [ -e "$OUT/${TAG}_best.pt" ]; then mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M)"; fi
mkdir -p "$OUT"

$PY -m src.engine.train_student --config configs/student_kd.yaml --teacher-ckpt "$TEACHER" --tag "$TAG" \
    --opt model.quality_mode=labelfree out_dir=$OUT train.epochs=$EPOCHS > logs/labelfree_student_train.log 2>&1 &
TRAIN_PID=$!
echo "[lfstudent] training pid $TRAIN_PID"

# Watchdog: (a) out_dir override must have worked, (b) epoch 1 must show a sane teacher agreement / QWK.
LOG="$OUT/kd_${TAG}.log"
elapsed=0; checked=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60; elapsed=$((elapsed+60))
  if [ ! -f "$LOG" ] && [ "$elapsed" -gt 600 ]; then
    echo "[lfstudent] FAILED: $LOG not created (out_dir override not honoured?)"; kill "$TRAIN_PID"; echo FAILED_OUTDIR > "$OUT/FAILED"; exit 2
  fi
  if [ "$checked" -eq 0 ] && [ -f "$LOG" ] && grep -q "epoch 1:" "$LOG"; then
    line=$(grep -m1 "epoch 1:" "$LOG")
    agree=$(echo "$line" | sed -n 's/.*teacher-agreement=\([0-9.]*\).*/\1/p')
    qwk=$(echo "$line" | sed -n 's/.*qwk=\([0-9.-]*\).*/\1/p')
    echo "[lfstudent] epoch1 check: agreement=$agree val_qwk=$qwk"
    if awk "BEGIN{exit !(${agree:-0} < 0.85 || ${qwk:-0} <= 0.05)}"; then
      echo "[lfstudent] FAILED: student is not learning (teacher probably not loaded correctly)"; kill "$TRAIN_PID"; echo FAILED_EPOCH1 > "$OUT/FAILED"; exit 3
    fi
    checked=1
  fi
done
wait "$TRAIN_PID"; rc=$?
echo "[lfstudent $(date '+%m-%d %H:%M')] training exited rc=$rc"
[ -f "$OUT/${TAG}_best.pt" ] || { echo "[lfstudent] no best checkpoint, aborting"; exit 4; }

# Evaluate the new student on the in-domain test split (temperature-scaled, tagged so nothing is overwritten).
$PY -m src.engine.evaluate --config configs/student_kd.yaml --ckpt "$OUT/${TAG}_best.pt" --split test \
    --calibrate-temperature --tag "$TAG" --student > logs/labelfree_student_eval.log 2>&1
echo "[lfstudent] eval rc=$?"

{
  echo "label-free student finished $(date)"
  echo "--- last epochs ---"; tail -n 5 "$LOG"
  echo "--- eval log tail ---"; tail -n 15 logs/labelfree_student_eval.log
} > logs/LFSTUDENT_SUMMARY.txt
tar -czf "backups/auto_lfstudent_$(date +%m%d_%H%M).tgz" "$OUT"/*.log "$OUT"/*.json runs/student/*"$TAG"* 2>/dev/null
echo "[lfstudent $(date '+%m-%d %H:%M')] DONE"
