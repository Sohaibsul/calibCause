#!/usr/bin/env bash
# Resume-safe, duplicate-safe queue for the remaining LOEO ablation runs
# (reviewer comment 1). Re-running this exact script after ANY interruption
# (shutdown, crash, reboot) continues where it stopped:
#   * run already evaluated (report json exists)  -> skipped
#   * `<tag>_last.pt` exists                      -> resumed from that epoch
#   * otherwise                                   -> started from scratch
# A lock file guarantees only ONE copy of the queue can run at a time, so a
# repeated launch (e.g. pasted twice) exits immediately instead of creating a
# duplicate training process.
# Usage:   nohup bash scripts/run_ablation_queue.sh > logs/ablation_queue.log 2>&1 &
# Dry run: DRY=1 bash scripts/run_ablation_queue.sh   (prints what it WOULD do)
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

exec 9>/tmp/calibcause_queue.lock
if ! flock -n 9; then
  echo "[queue] another queue is already running - exiting (no duplicate started)"
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DRY="${DRY:-0}"

# run_one <name> <config> <out_dir> <tag> <train envs...> -- <extra train args> -- <extra eval args>
run_one() {
  local name="$1" cfg="$2" out="$3" tag="$4"; shift 4
  local envs=() tr_extra=() ev_extra=() part=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then part=$((part + 1)); continue; fi
    case $part in 0) envs+=("$a");; 1) tr_extra+=("$a");; 2) ev_extra+=("$a");; esac
  done
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then echo "[queue] SKIP   $name (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); echo "[queue] RESUME $name from $last"
  else echo "[queue] START  $name"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_stage2 --config "$cfg" "${tr_extra[@]}" \
      --train-envs "${envs[@]}" --val-envs "${envs[@]}" --tag "$tag" "${resume[@]}" \
      >> "logs/${tag}.log" 2>&1 || { echo "[queue] FAILED training $name (see logs/${tag}.log)"; return 1; }
  python -m src.engine.evaluate --config "$cfg" --ckpt "$best" --split test \
      "${ev_extra[@]}" --tag "$tag" >> "logs/${tag}.log" 2>&1 \
      || { echo "[queue] FAILED evaluate $name"; return 1; }
  echo "[queue] DONE   $name"
}

# Priority order (most informative first): held-out EyePACS first, then the
# other two folds for the no-causal ablation, then the label-free folds.
run_one "labelfree / heldout eyepacs" configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_eyepacs ddr aptos -- --quality-mode labelfree -- --quality-mode labelfree
run_one "nocausal  / heldout ddr"     configs/ablation_nocausal.yaml  runs/ablation_nocausal \
  nocausal_loeo_heldout_ddr eyepacs aptos -- --no-causal --
run_one "nocausal  / heldout aptos"   configs/ablation_nocausal.yaml  runs/ablation_nocausal \
  nocausal_loeo_heldout_aptos eyepacs ddr -- --no-causal --
run_one "labelfree / heldout ddr"     configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_ddr eyepacs aptos -- --quality-mode labelfree -- --quality-mode labelfree
run_one "labelfree / heldout aptos"   configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_aptos eyepacs ddr -- --quality-mode labelfree -- --quality-mode labelfree
echo "[queue] all runs finished or skipped"
