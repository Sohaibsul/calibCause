#!/usr/bin/env bash
# Distil a student from the label-free teacher (resume-safe). Output: runs/student_labelfree, tag student_lf.
# Re-running this script after a shutdown resumes from student_lf_last.pt.
set -u
cd "$HOME/Grading_implementation" || exit 1
PY="$PWD/.venv/bin/python"
TEACHER=runs/ablation_labelfree/labelfree_best.pt
OUT=runs/student_labelfree
TAG=student_lf
EPOCHS=15
MIN_GPU_MB=3500
MIN_RAM_MB=6144
MAX_TRIES=4
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

# Protect the existing Full-model student.
[ -f runs/student/student_best.pt ] && cp --update=none runs/student/student_best.pt backups/student_full_best_backup.pt 2>/dev/null
# Resume if a resumable checkpoint exists; otherwise start clean (move any old partial output aside).
if [ -f "$OUT/${TAG}_last.pt" ]; then
  echo "[lfstudent] resumable checkpoint found, will resume"
elif [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  mv "$OUT" "${OUT}_old_$(date +%m%d_%H%M)"
fi
mkdir -p "$OUT"; rm -f "$OUT/FAILED"

attempt=0; rc=1
while [ "$attempt" -lt "$MAX_TRIES" ]; do
  attempt=$((attempt+1))
  wait_for_resources
  echo "[lfstudent $(date '+%m-%d %H:%M')] attempt $attempt/$MAX_TRIES"
  $PY -m src.engine.train_student_v2 --config configs/student_kd.yaml --teacher-ckpt "$TEACHER" --tag "$TAG" --resume \
      --opt model.quality_mode=labelfree out_dir=$OUT train.epochs=$EPOCHS >> logs/labelfree_student_train.log 2>&1 &
  TRAIN_PID=$!
  checked=0
  # Watchdog: epoch 1 must show a sane teacher agreement / QWK (only checked once per launcher run).
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    sleep 60
    if [ "$checked" -eq 0 ] && [ -f "$LOG" ] && grep -q "epoch 1:" "$LOG"; then
      line=$(grep -m1 "epoch 1:" "$LOG")
      agree=$(echo "$line" | sed -n 's/.*teacher-agreement=\([0-9.]*\).*/\1/p')
      qwk=$(echo "$line" | sed -n 's/.*qwk=\([0-9.-]*\).*/\1/p')
      echo "[lfstudent] epoch1 check: agreement=$agree val_qwk=$qwk"
      if awk "BEGIN{exit !(${agree:-0} < 0.85 || ${qwk:-0} <= 0.05)}"; then
        echo "[lfstudent] FAILED: student is not learning"; kill "$TRAIN_PID"; echo FAILED_EPOCH1 > "$OUT/FAILED"; exit 2
      fi
      checked=1
    fi
  done
  wait "$TRAIN_PID"; rc=$?
  echo "[lfstudent $(date '+%m-%d %H:%M')] training exited rc=$rc"
  [ "$rc" -eq 0 ] && break
  [ "$rc" -eq 3 ] && { echo "[lfstudent] FAILED: teacher problem (see $LOG)"; echo FAILED_TEACHER > "$OUT/FAILED"; exit 3; }
  sleep 120   # crash (e.g. out of memory): retry, resuming from the last epoch
done
[ "$rc" -eq 0 ] || { echo "[lfstudent] giving up after $MAX_TRIES attempts"; exit 4; }
[ -f "$OUT/${TAG}_best.pt" ] || { echo "[lfstudent] no best checkpoint"; exit 4; }

# In-domain test evaluation of the new student (tagged, nothing is overwritten).
$PY -m src.engine.evaluate --config configs/student_kd.yaml --ckpt "$OUT/${TAG}_best.pt" --split test \
    --calibrate-temperature --tag "$TAG" --student > logs/labelfree_student_eval.log 2>&1
echo "[lfstudent] eval rc=$?"
{ echo "label-free student finished $(date)"; echo "--- last epochs ---"; grep "epoch" "$LOG" | tail -n 5
  echo "--- eval log tail ---"; tail -n 15 logs/labelfree_student_eval.log; } > logs/LFSTUDENT_SUMMARY.txt
tar -czf "backups/auto_lfstudent_$(date +%m%d_%H%M).tgz" "$OUT"/*.log "$OUT"/*.json runs/student/*"$TAG"* 2>/dev/null
echo "[lfstudent $(date '+%m-%d %H:%M')] DONE"
