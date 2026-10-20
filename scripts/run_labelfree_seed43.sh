#!/usr/bin/env bash
# Second-seed run of the label-free ablation on the held-out-EyePACS fold (seed 43),
# so that Full and Label-free both have two seeds for a fair comparison in the paper.
# Same config, flags and train/val envs as the seed-42 label-free LOEO run; only seed=43 differs.
#   tag: labelfree_loeo_heldout_eyepacs_seed43   (~8 h)
#
# Scheduling and safety:
#   1. It waits (up to 15 h) on the shared GPU queue lock, i.e. for run_seed43.sh (Full seed 43) to finish.
#   2. After that it still waits until the machine really has room: host RAM available >= MIN_RAM_MB and
#      GPU memory free >= MIN_GPU_MB, on 3 consecutive checks (this accounts for the thesis jobs).
#   3. While training, if host RAM stays below PAUSE_RAM_MB it stops ONLY this training (exact --tag match),
#      waits for free resources again and resumes from _last.pt. It never touches any other process.
# Safe to launch ONCE (a second copy exits). After any shutdown: relaunch run_seed43.sh FIRST, then this
# script. Finished steps are skipped, an unfinished training resumes from _last.pt.
#
#   nohup nice -n 10 bash scripts/run_labelfree_seed43.sh > logs/labelfree_seed43.log 2>&1 &
#   DRY=1 bash scripts/run_labelfree_seed43.sh        # prints the plan, runs nothing
set -u
cd "$(dirname "$0")/.."
mkdir -p logs backups

exec 5>/tmp/calibcause_labelfree43_launcher.lock
if ! flock -n 5; then
  echo "[lf43] already launched - exiting (no duplicate started)"
  exit 0
fi

DRY="${DRY:-0}"
SEED=43
MIN_RAM_MB="${MIN_RAM_MB:-12288}"      # start only with >= 12 GB host RAM available
MIN_GPU_MB="${MIN_GPU_MB:-4500}"       # start only with >= 4.5 GB GPU memory free (job needs ~2.1 GB)
PAUSE_RAM_MB="${PAUSE_RAM_MB:-2560}"   # pause this training if host RAM available stays < 2.5 GB
WAIT_SLEEP="${WAIT_SLEEP:-300}"        # seconds between checks while resources are short
STREAK_SLEEP="${STREAK_SLEEP:-60}"     # seconds between the 3 confirming checks
GUARD_SLEEP="${GUARD_SLEEP:-20}"       # seconds between RAM checks while training
say() { echo "[lf43 $(date '+%m-%d %H:%M')] $*"; }

ram_avail_mb() { awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo; }
gpu_free_mb()  { local v; v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '); echo "${v:-0}"; }

wait_for_resources() {
  local streak=0 n=0 r g
  while true; do
    r=$(ram_avail_mb); g=$(gpu_free_mb)
    if [ "$r" -ge "$MIN_RAM_MB" ] && [ "$g" -ge "$MIN_GPU_MB" ]; then
      streak=$((streak + 1))
      if [ "$streak" -ge 3 ]; then say "resources free: RAM ${r} MB available, GPU ${g} MB free"; return 0; fi
      sleep "$STREAK_SLEEP"
    else
      streak=0; n=$((n + 1))
      if [ $((n % 6)) -eq 1 ]; then
        say "waiting for free resources: RAM ${r} MB available (need ${MIN_RAM_MB}), GPU ${g} MB free (need ${MIN_GPU_MB})"
      fi
      sleep "$WAIT_SLEEP"
    fi
  done
}

exec 9>/tmp/calibcause_queue.lock
if [ "$DRY" != "1" ]; then
  say "waiting for the GPU queue lock (Full seed 43 finishes first)"
  if ! flock -w 54000 9; then
    say "could not get the GPU queue lock within 15 h - exiting"
    exit 1
  fi
  say "GPU queue lock acquired"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_labelfree() {   # label-free ablation, held-out EyePACS, seed 43
  local tag="labelfree_loeo_heldout_eyepacs_seed${SEED}" out="runs/ablation_labelfree" lg
  lg="logs/${tag}.log"
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then say "SKIP   labelfree seed${SEED} (report exists)"; return 0; fi
  if [ "$DRY" = "1" ]; then
    if [ -f "$last" ]; then say "RESUME labelfree seed${SEED}"; else say "START  labelfree seed${SEED}"; fi
    echo "        (dry run, nothing launched)"; return 0
  fi
  local attempts=0 max_attempts=30 tp low paused
  while [ "$attempts" -lt "$max_attempts" ]; do
    attempts=$((attempts + 1))
    wait_for_resources
    local resume=()
    if [ -f "$last" ]; then resume=(--resume "$last"); say "RESUME labelfree seed${SEED}"; else say "START  labelfree seed${SEED}"; fi
    python -m src.engine.train_stage2 --config configs/ablation_labelfree.yaml --quality-mode labelfree \
        --train-envs ddr aptos --val-envs ddr aptos --opt seed=${SEED} --tag "$tag" "${resume[@]}" >> "$lg" 2>&1 &
    tp=$!; low=0; paused=0
    while kill -0 "$tp" 2>/dev/null; do
      sleep "$GUARD_SLEEP"
      if [ "$(ram_avail_mb)" -lt "$PAUSE_RAM_MB" ]; then low=$((low + 1)); else low=0; fi
      if [ "$low" -ge 2 ]; then
        say "PAUSE  labelfree: host RAM low ($(ram_avail_mb) MB available) - stopping only this training, will resume from _last.pt"
        pkill -TERM -f -- "--tag $tag"; sleep 10
        pkill -KILL -f -- "--tag $tag" 2>/dev/null
        wait "$tp" 2>/dev/null; paused=1; break
      fi
    done
    if [ "$paused" = "1" ]; then sleep 120; continue; fi
    wait "$tp" || { say "FAILED training labelfree seed${SEED} (see $lg)"; return 1; }
    break
  done
  if [ ! -f "$best" ]; then say "no finished training after $attempts attempts - giving up"; return 1; fi
  python -m src.engine.evaluate --config configs/ablation_labelfree.yaml --ckpt "$best" --split test \
      --quality-mode labelfree --tag "$tag" >> "$lg" 2>&1 || { say "FAILED evaluate labelfree seed${SEED}"; return 1; }
  say "DONE   labelfree seed${SEED}"
}

backup() {
  [ "$DRY" = "1" ] && return 0
  local ts; ts=$(date +%Y%m%d_%H%M)
  find runs -type f \( -name 'report_*.json' -o -name 'calib_*.json' -o -name 'history_*.json' \) -print0 \
    | tar --null -czf "backups/auto_results_$ts.tar.gz" -T - 2>/dev/null
  find runs -type f -name 'preds_*.npz' -print0 \
    | tar --null -czf "backups/auto_preds_$ts.tar.gz" -T - 2>/dev/null
  say "backup written: backups/auto_results_$ts.tar.gz (+ preds)"
}

run_labelfree; backup

if [ "$DRY" != "1" ]; then
  P=()
  for spec in "plain_s42=runs/baseline/preds_baseline2_loeo_heldout_eyepacs_test.npz" \
              "plain_s43=runs/baseline/preds_baseline2_loeo_heldout_eyepacs_seed${SEED}_test.npz" \
              "full_s42=runs/stage2/preds_loeo_heldout_eyepacs_test.npz" \
              "full_s43=runs/stage2/preds_loeo_heldout_eyepacs_seed${SEED}_test.npz" \
              "labelfree_s42=runs/ablation_labelfree/preds_labelfree_loeo_heldout_eyepacs_test.npz" \
              "labelfree_s43=runs/ablation_labelfree/preds_labelfree_loeo_heldout_eyepacs_seed${SEED}_test.npz"; do
    [ -f "${spec#*=}" ] && P+=(--model "$spec")
  done
  if [ "${#P[@]}" -ge 12 ]; then
    python scripts/paired_bootstrap.py --fold eyepacs "${P[@]}" --out runs/bootstrap_labelfree_seed43_eyepacs.json \
        > logs/LABELFREE43_SUMMARY.txt 2>&1
    say "summary written to logs/LABELFREE43_SUMMARY.txt"
  else
    say "not all six runs are finished - summary skipped"
  fi
fi
say "all steps finished or skipped"
