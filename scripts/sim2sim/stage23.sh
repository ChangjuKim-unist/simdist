#!/bin/bash
# Design doc 23: weight interpolation (alpha) and terminal value weight (w) on go1_paper and its SimDist adaptations.
# 45 runs (6 seeds x 3 speeds x 5000 steps each) on 4 GPUs, ~10.5 h. Jobs are pulled from a shared queue.
#   usage (gpu1): tmux new -d -s s23 'bash scripts/sim2sim/stage23.sh > results/stage23_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage23; mkdir -p $OUT
BASE=go1_paper; M=go1_paper_very_weak_motor_dyn_b45000_s42; F=go1_paper_friction0.15_dyn_b45000_s42
Q=$OUT/queue.txt; LOCK=$OUT/queue.lock
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
: > $Q
for c in motor0.4 friction0.15 nominal; do
  echo "base_a0|$BASE|$c|" >> $Q
  for d in M F; do ck=$([ $d = M ] && echo $M || echo $F)
    echo "${d}_a1|$ck|$c|" >> $Q
    for a in 0.25 0.5 0.75; do echo "${d}_a$a|$ck|$c|model.interp_from=$BASE model.interp_alpha=$a" >> $Q; done
  done
  for w in 0 0.5; do
    echo "base_w$w|$BASE|$c|control.value_weight=$w" >> $Q
    echo "M_w$w|$M|$c|control.value_weight=$w" >> $Q
    echo "F_w$w|$F|$c|control.value_weight=$w" >> $Q
  done
done
echo "### stage23 start $(date) jobs $(wc -l < $Q)"
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name ckpt cond extra <<< "$job"
    echo "== $name $cond gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 \
      sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv \
      output.model_name=$name output.condition=$cond $extra --headless 2>&1 | grep -E "Traceback|rror:|Interpolated" | head -3
  done; }
for g in 0 1 2 3; do worker $g & sleep 20; done
wait
echo "### stage23 done $(date)"
