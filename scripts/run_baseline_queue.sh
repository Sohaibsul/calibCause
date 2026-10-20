#!/usr/bin/env bash
# Takeover queue: replaces the tail of run_ablation_queue.sh.
#
# Why: label-free held-out-APTOS (~11-16 h) adds little (label-free was within
# noise of the full model on the EyePACS fold), whereas the plain ConvNeXt-Tiny
# baseline on the DDR and APTOS folds is what reviewer comment 2 needs.
#
# What it does, in order (all resume-safe, same rules as run_ablation_queue.sh):
#   0. If the old ablation queue is still running, wait until its
#      "labelfree / heldout ddr" run has produced its report, then stop the old
#      queue BEFORE it starts "labelfree / heldout aptos" (kills only the old
#      queue shell and a labelfree-aptos training, nothing else).
#   1. labelfree / heldout ddr          (skipped if its report exists)
#   2. baseline  / heldout ddr
#   3. baseline  / heldout aptos
#   4. labelfree / heldout aptos        (LAST; optional - stop it if time is short)
#
# Safe to launch exactly once; a second copy exits immediately (launcher lock).
# After a shutdown, launch THIS script (not run_ablation_queue.sh) to continue.
# Usage:   nohup bash scripts/run_baseline_queue.sh > logs/baseline_queue.log 2>&1 &
# Dry run: DRY=1 bash scripts/run_baseline_queue.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

# Launcher guard: only one copy of THIS script, even while it is waiting.
exec 8>/tmp/calibcause_baseq_launcher.lock
if ! flock -n 8; then
  echo "[queue2] already launched - exiting (no duplicate started)"
  exit 0
fi

DRY="${DRY:-0}"
OLD_PAT="bash scripts/run_ablation_queue.sh"
LF_DDR_REPORT="runs/ablation_labelfree/report_labelfree_loeo_heldout_ddr_test.json"

# ---- 0. hand-over from the old queue -------------------------------------
if [ "$DRY" != "1" ] && pgrep -f "$OLD_PAT" >/dev/null; then
  echo "[queue2] old ablation queue is running; waiting for labelfree/heldout ddr to finish"
  while [ ! -f "$LF_DDR_REPORT" ]; do
    pgrep -f "$OLD_PAT" >/dev/null || break   # old queue ended by itself
    sleep 15
  done
  if pgrep -f "$OLD_PAT" >/dev/null; then
    sleep 5                                    # let evaluate exit cleanly
    echo "[queue2] labelfree/heldout ddr done -> stopping old queue (before labelfree/aptos)"
    pkill -TERM -f "$OLD_PAT"                  # 1) queue shell first: it can't start anything new
    sleep 2
    pkill -TERM -f -- "--tag labelfree_loeo_heldout_aptos" 2>/dev/null || true   # 2) a just-started aptos run
    sleep 5
  fi
fi

# Wait (max 10 min) until nobody holds the shared GPU-queue lock any more.
exec 9>/tmp/calibcause_queue.lock
if [ "$DRY" != "1" ] && ! flock -w 600 9; then
  echo "[queue2] could not get the queue lock (something else is still running) - exiting"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- helpers -------------------------------------------------------------
# run_one <name> <config> <out_dir> <tag> <train envs...> -- <extra train args> -- <extra eval args>
run_one() {
  local name="$1" cfg="$2" out="$3" tag="$4"; shift 4
  local envs=() tr_extra=() ev_extra=() part=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then part=$((part + 1)); continue; fi
    case $part in 0) envs+=("$a");; 1) tr_extra+=("$a");; 2) ev_extra+=("$a");; esac
  done
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then echo "[queue2] SKIP   $name (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); echo "[queue2] RESUME $name from $last"
  else echo "[queue2] START  $name"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_stage2 --config "$cfg" "${tr_extra[@]}" \
      --train-envs "${envs[@]}" --val-envs "${envs[@]}" --tag "$tag" "${resume[@]}" \
      >> "logs/${tag}.log" 2>&1 || { echo "[queue2] FAILED training $name (see logs/${tag}.log)"; return 1; }
  python -m src.engine.evaluate --config "$cfg" --ckpt "$best" --split test \
      "${ev_extra[@]}" --tag "$tag" >> "logs/${tag}.log" 2>&1 \
      || { echo "[queue2] FAILED evaluate $name"; return 1; }
  echo "[queue2] DONE   $name"
}

# run_baseline <name> <tag> <train envs...>   (plain ConvNeXt-Tiny, same 30-epoch recipe)
run_baseline() {
  local name="$1" tag="$2"; shift 2
  local envs=("$@") out="runs/baseline"
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then echo "[queue2] SKIP   $name (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); echo "[queue2] RESUME $name from $last"
  else echo "[queue2] START  $name"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_baseline --config configs/baseline.yaml \
      --train-envs "${envs[@]}" --val-envs "${envs[@]}" --tag "$tag" "${resume[@]}" \
      >> "logs/${tag}.log" 2>&1 || { echo "[queue2] FAILED training $name (see logs/${tag}.log)"; return 1; }
  python -m src.engine.evaluate --config configs/baseline.yaml --ckpt "$best" --student \
      --split test --tag "$tag" >> "logs/${tag}.log" 2>&1 \
      || { echo "[queue2] FAILED evaluate $name"; return 1; }
  echo "[queue2] DONE   $name"
}

# ---- run list --------------------------------------------------------------
run_one "labelfree / heldout ddr"   configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_ddr eyepacs aptos -- --quality-mode labelfree -- --quality-mode labelfree
run_baseline "baseline2 / heldout eyepacs" baseline2_loeo_heldout_eyepacs ddr aptos
run_baseline "baseline2 / heldout ddr"     baseline2_loeo_heldout_ddr     eyepacs aptos
run_baseline "baseline2 / heldout aptos"   baseline2_loeo_heldout_aptos   eyepacs ddr
run_one "labelfree / heldout aptos" configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_aptos eyepacs ddr -- --quality-mode labelfree -- --quality-mode labelfree
echo "[queue2] all runs finished or skipped"
