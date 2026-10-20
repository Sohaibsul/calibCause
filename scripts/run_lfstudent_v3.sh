#!/usr/bin/env bash
# Label-free student distillation, resume-safe, with automatic lower-lr retries if the student collapses.
# Output: runs/student_labelfree, tag student_lf. Re-running after a shutdown resumes from student_lf_last.pt.
set -u
cd "$HOME/Grading_implementation" || exit 1
PY="$PWD/.venv/bin/python"
TEACHER=runs/ablation_labelfree/labelfree_best.pt
OUT=runs/student_labelfree
TAG=student_lf
EPOCHS=15
MIN_GPU_MB=3500
MIN_RAM_MB=6144
MAX_TRIES=5
LRS=("" "train.lr=1.5e-4" "train.lr=1.0e-4")   # tried in order; the next one is used only after a collapse (exit 5)
LOG="$OUT/kd_${TAG}.log"
mkdir -p logs backups

exec 6>/tmp/calibcause_lfstudent_launcher.lock
flock -n 6 || { echo "[lfstudent] already running, exiting"; exit 0; }
exec 9>/tmp/calibcause_queue.lock
echo "[lfstudent $(date '+%m-%d %H:%M')] waiting for queue lock"
flock -w 54000 9 || { echo "[lfstudent] queue lock timeout"; exit 1; }
[ -f "$TEACHER" ] || { echo "[lfstudent] teacher checkpoint missing: $TEACHER"; exit 1; }

wait_for_resources() {
  local ok=0 gpu_free ram_av
  while [ "$ok" -lt 3 ]; do
    gpu_free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}' | head -1)
    ram_av=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
    if [ "${gpu_free:-0}" -ge "$MIN_GPU_MB" ] && [ "${ram_av:-0}" -ge "$MIN_RAM_MB" ]; then ok=$((ok+1)); else ok=0; fi
    [ "$ok" -lt 3 ] && sleep 30
  done
  echo "[lfstudent $(date '+%m-%d %H:%M')] resources OK (gpu_free=${gpu_free}MB ram_avail=${ram_av}MB)"
}

[ -f runs/student/student_best.pt ] && cp --update=none runs/student/student_best.pt backups/student_full_best_backup.pt 2>/dev/null
if [ -f "$OUT/${TAG}_last.pt" ] && [ ! -f "$OUT/FAILED" ]; then
  echo "[lfstudent] resumable checkpoint found, will resume"
elif [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M)"
fi
mkdir -p "$OUT"; rm -f "$OUT/FAILED"

attempt=0; lr_idx=0; rc=1; first=1
while [ "$attempt" -lt "$MAX_TRIES" ]; do
  attempt=$((attempt+1))
  wait_for_resources
  LR_OPT="${LRS[$lr_idx]}"
  SKIP=""; [ "$first" -eq 0 ] && SKIP="--skip-teacher-check"
  echo "[lfstudent $(date '+%m-%d %H:%M')] attempt $attempt/$MAX_TRIES  lr_override='${LR_OPT}'"
  $PY -m src.engine.train_student_v2 --config configs/student_kd.yaml --teacher-ckpt "$TEACHER" --tag "$TAG" --resume $SKIP \
      --opt model.quality_mode=labelfree out_dir=$OUT train.epochs=$EPOCHS $LR_OPT >> logs/labelfree_student_train.log 2>&1 &
  TRAIN_PID=$!
  first=0; checked=0
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    if [ "$checked" -eq 0 ] && [ -f "$LOG" ] && grep -a -q "epoch 1:" "$LOG"; then
      line=$(grep -a "epoch 1:" "$LOG" | tail -n 1)
      agree=$(echo "$line" | sed -n 's/.*teacher-agreement=\([0-9.]*\).*/\1/p')
      qwk=$(echo "$line" | sed -n 's/.*qwk=\([0-9.-]*\).*/\1/p')
      echo "[lfstudent] epoch1 check: agreement=$agree val_qwk=$qwk"
      if awk "BEGIN{exit !(${agree:-0} < 0.80 || ${qwk:-0} <= 0.05)}"; then
        echo "[lfstudent] FAILED at epoch 1 (student not learning)"; kill "$TRAIN_PID"; echo FAILED_EPOCH1 > "$OUT/FAILED"; exit 2
      fi
      checked=1
    fi
  done
  wait "$TRAIN_PID"; rc=$?
  echo "[lfstudent $(date '+%m-%d %H:%M')] training exited rc=$rc"
  [ "$rc" -eq 0 ] && break
  [ "$rc" -eq 3 ] && { echo "[lfstudent] FAILED: teacher problem (see $LOG)"; echo FAILED_TEACHER > "$OUT/FAILED"; exit 3; }
  if [ "$rc" -eq 5 ]; then
    lr_idx=$((lr_idx+1))
    [ "$lr_idx" -ge "${#LRS[@]}" ] && { echo "[lfstudent] collapsed at every learning rate, giving up"; echo FAILED_COLLAPSE > "$OUT/FAILED"; exit 5; }
    echo "[lfstudent] student collapsed, retrying with ${LRS[$lr_idx]}"
  fi
  sleep 60
done
[ "$rc" -eq 0 ] || { echo "[lfstudent] giving up after $MAX_TRIES attempts"; exit 4; }
[ -f "$OUT/${TAG}_best.pt" ] || { echo "[lfstudent] no best checkpoint"; exit 4; }

$PY -m src.engine.evaluate --config configs/student_kd.yaml --ckpt "$OUT/${TAG}_best.pt" --split test \
    --calibrate-temperature --tag "$TAG" --student > logs/labelfree_student_eval.log 2>&1
echo "[lfstudent] eval rc=$?"
{ echo "label-free student finished $(date)"; echo "--- last epochs ---"; grep -a "epoch" "$LOG" | tail -n 5
  echo "--- eval log tail ---"; tail -n 15 logs/labelfree_student_eval.log; } > logs/LFSTUDENT_SUMMARY.txt
tar -czf "backups/auto_lfstudent_$(date +%m%d_%H%M).tgz" "$OUT"/*.log "$OUT"/*.json runs/student/*"$TAG"* 2>/dev/null
echo "[lfstudent $(date '+%m-%d %H:%M')] DONE"
