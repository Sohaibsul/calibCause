#!/usr/bin/env bash
# Second-seed check on the held-out-EyePACS fold (reviewer: "single seed").
# Trains the SAME two models again with seed=43 and nothing else changed:
#   1. plain ConvNeXt-T (fixed recipe)   tag baseline2_loeo_heldout_eyepacs_seed43   (~4-5 h)
#   2. full CalibCause (stage2_moe.yaml) tag loeo_heldout_eyepacs_seed43              (~8 h)
# then writes a paired-bootstrap summary of all four runs (seed 42 + seed 43) to logs/SEED43_SUMMARY.txt.
#
# Safe to launch ONCE (a second copy exits). It waits (up to 4 h) for the GPU queue lock, i.e. for
# run_overnight.sh to finish, so it can be launched while that is still running. After any shutdown
# launch the same command again: finished steps are skipped, unfinished ones resume from _last.pt.
#
#   nohup nice -n 10 bash scripts/run_seed43.sh > logs/seed43.log 2>&1 &
#   DRY=1 bash scripts/run_seed43.sh        # prints the plan, runs nothing
set -u
cd "$(dirname "$0")/.."
mkdir -p logs backups

exec 6>/tmp/calibcause_seed43_launcher.lock
if ! flock -n 6; then
  echo "[seed43] already launched - exiting (no duplicate started)"
  exit 0
fi

DRY="${DRY:-0}"
WD_SLEEP="${WD_SLEEP:-60}"
SEED=43
say() { echo "[seed43 $(date '+%m-%d %H:%M')] $*"; }

exec 9>/tmp/calibcause_queue.lock
if [ "$DRY" != "1" ]; then
  say "waiting for the GPU queue lock (other CalibCause jobs finish first)"
  if ! flock -w 14400 9; then
    say "could not get the GPU queue lock within 4 h - exiting"
    exit 1
  fi
  say "GPU queue lock acquired"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

diverged() {   # diverged <logfile>
  local lg="$1"
  [ "$(grep -aEc 'epoch [0-9]+: loss=nan' "$lg" 2>/dev/null)" -ge 2 ] && return 0
  [ "$(grep -a 'val acc=' "$lg" 2>/dev/null | tail -n 3 | grep -c 'qwk=0\.0000')" -ge 3 ] && return 0
  return 1
}

run_plain() {   # plain ConvNeXt-T, seed 43
  local tag="baseline2_loeo_heldout_eyepacs_seed${SEED}" out="runs/baseline" lg
  lg="logs/${tag}.log"
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then say "SKIP   plain seed${SEED} (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); say "RESUME plain seed${SEED}"; else say "START  plain seed${SEED}"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_baseline --config configs/baseline.yaml \
      --train-envs ddr aptos --val-envs ddr aptos --opt seed=${SEED} --tag "$tag" "${resume[@]}" >> "$lg" 2>&1 &
  local tp=$!
  while kill -0 "$tp" 2>/dev/null; do
    sleep "$WD_SLEEP"
    if diverged "$lg"; then
      say "ABORT  plain seed${SEED}: diverged (NaN loss / QWK stuck at 0)"
      pkill -TERM -f -- "--tag $tag"; sleep 5; wait "$tp" 2>/dev/null
      return 1
    fi
  done
  wait "$tp" || { say "FAILED training plain seed${SEED} (see $lg)"; return 1; }
  python -m src.engine.evaluate --config configs/baseline.yaml --ckpt "$best" --student \
      --split test --tag "$tag" >> "$lg" 2>&1 || { say "FAILED evaluate plain seed${SEED}"; return 1; }
  say "DONE   plain seed${SEED}"
}

run_full() {   # full CalibCause, seed 43 (same config/flags as the seed-42 LOEO run)
  local tag="loeo_heldout_eyepacs_seed${SEED}" out="runs/stage2" lg
  lg="logs/${tag}.log"
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then say "SKIP   full seed${SEED} (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); say "RESUME full seed${SEED}"; else say "START  full seed${SEED}"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_stage2 --config configs/stage2_moe.yaml \
      --train-envs ddr aptos --val-envs ddr aptos --opt seed=${SEED} --tag "$tag" "${resume[@]}" >> "$lg" 2>&1 \
      || { say "FAILED training full seed${SEED} (see $lg)"; return 1; }
  python -m src.engine.evaluate --config configs/stage2_moe.yaml --ckpt "$best" --split test \
      --tag "$tag" >> "$lg" 2>&1 || { say "FAILED evaluate full seed${SEED}"; return 1; }
  say "DONE   full seed${SEED}"
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

run_plain; backup
run_full;  backup

if [ "$DRY" != "1" ]; then
  P=()
  for spec in "plain_s42=runs/baseline/preds_baseline2_loeo_heldout_eyepacs_test.npz" \
              "plain_s43=runs/baseline/preds_baseline2_loeo_heldout_eyepacs_seed${SEED}_test.npz" \
              "full_s42=runs/stage2/preds_loeo_heldout_eyepacs_test.npz" \
              "full_s43=runs/stage2/preds_loeo_heldout_eyepacs_seed${SEED}_test.npz"; do
    [ -f "${spec#*=}" ] && P+=(--model "$spec")
  done
  if [ "${#P[@]}" -ge 4 ]; then
    python scripts/paired_bootstrap.py --fold eyepacs "${P[@]}" --out runs/bootstrap_seed43_eyepacs.json \
        > logs/SEED43_SUMMARY.txt 2>&1
    say "summary written to logs/SEED43_SUMMARY.txt"
  else
    say "not enough finished runs for the summary (need >= 2 models)"
  fi
fi
say "all steps finished or skipped"
