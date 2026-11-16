#!/bin/bash
# Design doc 26 stage 1: why the planner falls (startup jump, mid-walk). 9 runs with per-step diagnostics, ~2 h on 4 GPUs.
#   usage (gpu1): GPU_LIST="0 1 2 3" tmux new -d -s s26 'bash scripts/sim2sim/stage26.sh > results/stage26_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage26; mkdir -p $OUT
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
Q=$OUT/queue.txt; LOCK=$OUT/queue.lock
PRIOR=control.prior_only=true

MO=go1_paper_very_weak_motor_dyn_b45000_s42
FR=go1_paper_friction0.15_dyn_b45000_s42
: > $Q
# name|checkpoint|friction|extra overrides ; planner runs first (long), then the policy-only runs (short)
echo "D1_paper_plan_off|go1_paper|0.15|" >> $Q
echo "D2_paper_plan_off_fric0.8|go1_paper|0.8|" >> $Q
echo "D3_paperFR_plan_off|$FR|0.15|" >> $Q
echo "D9_paperMO_plan_off|$MO|0.15|" >> $Q
echo "D5_paper_plan_r50|go1_paper|0.15|sim.cmd_ramp_steps=50" >> $Q
echo "D6_broad_plan_r50|go1_broadwm|0.15|sim.cmd_ramp_steps=50" >> $Q
echo "D4_paper_pol_off|go1_paper|0.15|$PRIOR" >> $Q
echo "D7_broad_pol_r50|go1_broadwm|0.15|$PRIOR sim.cmd_ramp_steps=50" >> $Q
echo "D8_paper_pol_r50|go1_paper|0.15|$PRIOR sim.cmd_ramp_steps=50" >> $Q
echo "### stage26 start $(date) jobs $(wc -l < $Q)"
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name ckpt fric extra <<< "$job"
    echo "== $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt \
      sim.friction=$fric sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 \
      sim.episode_length_s=100 'batch.speeds=[0.9]' 'batch.seeds=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]' \
      output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv output.falls_csv=/workspace/simdist/$OUT/falls_gpu$g.csv \
      output.diag_npz=/workspace/simdist/$OUT/diag_$name.npz \
      output.model_name=$name output.condition=friction$fric $extra --headless 2>&1 | grep -E "Traceback|rror:|falls written|diagnostics written" | head -4
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage26 done $(date)"
