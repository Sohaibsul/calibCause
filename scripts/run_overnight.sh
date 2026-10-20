#!/usr/bin/env bash
# One-shot "leave it running" pipeline for everything that is still open.
# Safe to launch ONCE (a second copy exits). After any shutdown/crash, launch the same
# command again: every step is skipped if done, or resumed from its last checkpoint.
#
#   nohup nice -n 10 bash scripts/run_overnight.sh > logs/overnight.log 2>&1 &
#   DRY=1 bash scripts/run_overnight.sh        # prints the plan, runs nothing
#
# Order (training steps are followed by their own post-hoc steps; no step can block the next):
#   0. hand-over: if run_baseline_queue.sh is still running, wait for baseline2/eyepacs to finish,
#      then stop that queue (waiting for the next epoch checkpoint if a run is mid-way)
#   1. baseline2 (plain ConvNeXt-T, fixed recipe) / held-out EyePACS      [training]
#   2. post-hoc temperature scaling (T fitted on TRAINING-site val only) for finished LOEO runs
#   3. baseline2 / held-out DDR                                            [training]
#   4. baseline2 / held-out APTOS                                          [training]
#   5. label-free / held-out APTOS (last: least informative)              [training]
#   6. summary table -> logs/FINAL_SUMMARY.txt, backups -> backups/
# Divergence watchdog: a baseline run is aborted (not retried) if its epoch loss is NaN twice or
# its validation QWK is exactly 0.0000 for three epochs in a row, so a broken run cannot burn hours.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs backups

exec 7>/tmp/calibcause_overnight_launcher.lock
if ! flock -n 7; then
  echo "[night] already launched - exiting (no duplicate started)"
  exit 0
fi

DRY="${DRY:-0}"
WD_SLEEP="${WD_SLEEP:-60}"          # watchdog polling interval in seconds
OLD_PAT="bash scripts/run_baseline_queue.sh"
EY_REPORT="runs/baseline/report_baseline2_loeo_heldout_eyepacs_test.json"
say() { echo "[night $(date '+%m-%d %H:%M')] $*"; }

# ---- 0. hand-over from the previous queue --------------------------------------
if [ "$DRY" != "1" ] && pgrep -f "$OLD_PAT" >/dev/null; then
  say "previous queue is running; waiting for baseline2 / heldout eyepacs to finish"
  while [ ! -f "$EY_REPORT" ]; do
    pgrep -f "$OLD_PAT" >/dev/null || break
    sleep 15
  done
  if pgrep -f "$OLD_PAT" >/dev/null; then
    sleep 5
    cur_tag=$(pgrep -af "src.engine.train_(baseline|stage2)" | grep -oE -- "--tag [A-Za-z0-9_]+" | head -n 1 | awk '{print $2}')
    if [ -n "${cur_tag:-}" ]; then
      last=$(ls runs/*/"${cur_tag}"_last.pt 2>/dev/null | head -n 1)
      if [ -n "$last" ]; then
        say "waiting for the next epoch checkpoint of $cur_tag, so no finished epoch is lost"
        t0=$(stat -c %Y "$last"); waited=0
        while [ "$(stat -c %Y "$last")" = "$t0" ] && [ $waited -lt 3000 ] \
              && pgrep -f -- "--tag $cur_tag" >/dev/null; do
          sleep 10; waited=$((waited + 10))
        done
        sleep 15
      fi
    fi
    say "stopping the previous queue"
    pkill -TERM -f "$OLD_PAT"; sleep 2
    [ -n "${cur_tag:-}" ] && pkill -TERM -f -- "--tag $cur_tag"
    sleep 8
  fi
fi

exec 9>/tmp/calibcause_queue.lock
if [ "$DRY" != "1" ] && ! flock -w 900 9; then
  say "could not get the GPU queue lock (something else is still running) - exiting"
  exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- helpers -----------------------------------------------------------------
diverged() {   # diverged <logfile>
  local lg="$1"
  [ "$(grep -aEc 'epoch [0-9]+: loss=nan' "$lg" 2>/dev/null)" -ge 2 ] && return 0
  [ "$(grep -a 'val acc=' "$lg" 2>/dev/null | tail -n 3 | grep -c 'qwk=0\.0000')" -ge 3 ] && return 0
  return 1
}

run_baseline() {   # run_baseline <name> <tag> <train envs...>
  local name="$1" tag="$2"; shift 2
  local envs=("$@") out="runs/baseline" lg="logs/${tag}.log"
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then say "SKIP   $name (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); say "RESUME $name from $last"; else say "START  $name"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_baseline --config configs/baseline.yaml \
      --train-envs "${envs[@]}" --val-envs "${envs[@]}" --tag "$tag" "${resume[@]}" >> "$lg" 2>&1 &
  local tp=$!
  while kill -0 "$tp" 2>/dev/null; do
    sleep "$WD_SLEEP"
    if diverged "$lg"; then
      say "ABORT  $name: training diverged (NaN loss / QWK stuck at 0) - killing it, moving on"
      pkill -TERM -f -- "--tag $tag"; sleep 5; wait "$tp" 2>/dev/null
      return 1
    fi
  done
  wait "$tp" || { say "FAILED training $name (see $lg)"; return 1; }
  python -m src.engine.evaluate --config configs/baseline.yaml --ckpt "$best" --student \
      --split test --tag "$tag" >> "$lg" 2>&1 || { say "FAILED evaluate $name"; return 1; }
  say "DONE   $name"
}

run_one() {   # run_one <name> <config> <out_dir> <tag> <train envs...> -- <train args> -- <eval args>
  local name="$1" cfg="$2" out="$3" tag="$4"; shift 4
  local envs=() tr_extra=() ev_extra=() part=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then part=$((part + 1)); continue; fi
    case $part in 0) envs+=("$a");; 1) tr_extra+=("$a");; 2) ev_extra+=("$a");; esac
  done
  local report="$out/report_${tag}_test.json" last="$out/${tag}_last.pt" best="$out/${tag}_best.pt"
  if [ -f "$report" ]; then say "SKIP   $name (report exists)"; return 0; fi
  local resume=()
  if [ -f "$last" ]; then resume=(--resume "$last"); say "RESUME $name from $last"; else say "START  $name"; fi
  if [ "$DRY" = "1" ]; then echo "        (dry run, nothing launched)"; return 0; fi
  python -m src.engine.train_stage2 --config "$cfg" "${tr_extra[@]}" \
      --train-envs "${envs[@]}" --val-envs "${envs[@]}" --tag "$tag" "${resume[@]}" \
      >> "logs/${tag}.log" 2>&1 || { say "FAILED training $name (see logs/${tag}.log)"; return 1; }
  python -m src.engine.evaluate --config "$cfg" --ckpt "$best" --split test \
      "${ev_extra[@]}" --tag "$tag" >> "logs/${tag}.log" 2>&1 || { say "FAILED evaluate $name"; return 1; }
  say "DONE   $name"
}

calib() {   # calib <name> <config> <out_dir> <tag> <heldout> [extra calibrate_loeo args]
  local name="$1" cfg="$2" out="$3" tag="$4" held="$5"; shift 5
  local ckpt="$out/${tag}_best.pt" preds="$out/preds_${tag}_test.npz" js="$out/calib_${tag}.json"
  if [ -f "$js" ]; then say "calib SKIP   $name (already done)"; return 0; fi
  if [ ! -f "$ckpt" ] || [ ! -f "$preds" ]; then say "calib SKIP   $name (checkpoint or predictions missing)"; return 0; fi
  if [ "$DRY" = "1" ]; then say "calib WOULD RUN $name"; return 0; fi
  python scripts/calibrate_loeo.py --config "$cfg" --ckpt "$ckpt" --test-preds "$preds" \
      --heldout "$held" --out "$js" "$@" >> "logs/calib_${tag}.log" 2>&1 \
      && say "calib DONE   $name" || say "calib FAILED $name (see logs/calib_${tag}.log) - continuing"
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

calib_all_finished() {
  for e in eyepacs ddr aptos; do
    calib "nocausal / $e"  configs/ablation_nocausal.yaml  runs/ablation_nocausal  "nocausal_loeo_heldout_$e"  "$e"
    calib "labelfree / $e" configs/ablation_labelfree.yaml runs/ablation_labelfree "labelfree_loeo_heldout_$e" "$e" --quality-mode labelfree
    calib "baseline2 / $e" configs/baseline.yaml           runs/baseline           "baseline2_loeo_heldout_$e" "$e" --student
  done
}

# ---- run list ----------------------------------------------------------------
run_baseline "baseline2 / heldout eyepacs" baseline2_loeo_heldout_eyepacs ddr aptos
calib_all_finished; backup
run_baseline "baseline2 / heldout ddr"     baseline2_loeo_heldout_ddr     eyepacs aptos
calib_all_finished; backup
run_baseline "baseline2 / heldout aptos"   baseline2_loeo_heldout_aptos   eyepacs ddr
calib_all_finished; backup
run_one "labelfree / heldout aptos" configs/ablation_labelfree.yaml runs/ablation_labelfree \
  labelfree_loeo_heldout_aptos eyepacs ddr -- --quality-mode labelfree -- --quality-mode labelfree
calib_all_finished; backup

if [ "$DRY" != "1" ]; then
  python scripts/show_loeo.py > logs/FINAL_SUMMARY.txt 2>&1
  say "summary written to logs/FINAL_SUMMARY.txt"
fi
say "all steps finished or skipped"
