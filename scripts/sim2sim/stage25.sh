#!/bin/bash
# Design doc 25: gentle-start check on friction 0.15 at 0.9 m/s (18 seeds). 10 runs, ~2 h on 4 GPUs.
#   usage (gpu1): GPU_LIST="0 1 2 3" tmux new -d -s s25 'bash scripts/sim2sim/stage25.sh > results/stage25_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage25; mkdir -p $OUT
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
Q=$OUT/queue.txt; LOCK=$OUT/queue.lock
PRIOR=control.prior_only=true
: > $Q
# planner runs first (long), then the policy-only runs (short)
echo "paper_plan_off|go1_paper|" >> $Q
echo "paper_plan_r50|go1_paper|sim.cmd_ramp_steps=50" >> $Q
echo "broad_plan_off|go1_broadwm|" >> $Q
echo "broad_plan_r50|go1_broadwm|sim.cmd_ramp_steps=50" >> $Q
echo "paper_plan_r1|go1_paper|sim.cmd_ramp_steps=1" >> $Q
echo "paper_pol_off|go1_paper|$PRIOR" >> $Q
echo "paper_pol_r50|go1_paper|$PRIOR sim.cmd_ramp_steps=50" >> $Q
echo "paper_pol_r1|go1_paper|$PRIOR sim.cmd_ramp_steps=1" >> $Q
echo "broad_pol_off|go1_broadwm|$PRIOR" >> $Q
echo "broad_pol_r50|go1_broadwm|$PRIOR sim.cmd_ramp_steps=50" >> $Q
echo "### stage25 start $(date) jobs $(wc -l < $Q)"
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name ckpt extra <<< "$job"
    echo "== $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt \
      sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 \
      sim.episode_length_s=100 'batch.speeds=[0.9]' 'batch.seeds=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]' \
      output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv output.falls_csv=/workspace/simdist/$OUT/falls_gpu$g.csv \
      output.model_name=$name output.condition=friction0.15 $extra --headless 2>&1 | grep -E "Traceback|rror:|falls written" | head -3
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage25 done $(date)"
