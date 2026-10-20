#!/bin/bash
# Read-only status of the ablation + baseline queues. Does not launch or touch anything.
cd ~/Grading_implementation
echo "=== $(date '+%a %d %b %H:%M') ==="
pgrep -f "bash scripts/run_ablation_queue.sh" >/dev/null && echo "old ablation queue: RUNNING" || echo "old ablation queue: not running"
pgrep -f "bash scripts/run_baseline_queue.sh" >/dev/null && echo "baseline queue:     RUNNING" || echo "baseline queue:     not running"
echo
printf "%-34s %s\n" "RUN" "STATUS"
while read -r name dir; do
  rep="runs/$dir/report_${name}_test.json"
  log="logs/${name}.log"
  if [ -f "$rep" ]; then
    st="DONE"
  elif [ -f "$log" ]; then
    st="RUNNING: $(tail -c 600 "$log" | tr '\r' '\n' | grep -a 'epoch [0-9]*/' | tail -n 1 | cut -c1-110)"
  else
    st="WAITING"
  fi
  printf "%-34s %s\n" "$name" "$st"
done <<'LIST'
nocausal_loeo_heldout_ddr ablation_nocausal
nocausal_loeo_heldout_aptos ablation_nocausal
labelfree_loeo_heldout_ddr ablation_labelfree
baseline2_loeo_heldout_eyepacs baseline
baseline2_loeo_heldout_ddr baseline
baseline2_loeo_heldout_aptos baseline
labelfree_loeo_heldout_aptos ablation_labelfree
LIST
echo
nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu --format=csv,noheader
echo
python scripts/show_loeo.py
