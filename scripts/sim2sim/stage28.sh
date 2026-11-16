#!/bin/bash
# Design doc 28 (docs/design28_planner_motion_source_ko.md): where does the planner's unnatural motion come from?
#   arms: S4 (4x samples + elites), Sq (1/4 samples + elites), V0 (terminal value weight 0); base and policy reuse design 27.
#   broad model on normal floor (2000 steps) and ice (5000 steps), paper model on normal floor (2000 steps).
#   S4 runs 4 robots per process (memory), 5 processes per setting; the merge happens in stage28_analyze.py.
#   usage (gpu1): tmux new -d -s s28 'bash scripts/sim2sim/stage28.sh > results/stage28_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage28; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
COMMON="sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.episode_length_s=100 sim.cmd_ramp_steps=50 batch.speeds=[0.9]"
ALL="11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28"
PARTS=("11,12,13,14" "15,16,17,18" "19,20,21,22" "23,24,25,26" "27,28")
S4="control.num_samples=1800 control.num_elites=256"
SQ="control.num_samples=112 control.num_elites=16"
V0="control.value_weight=0.0"
FILT="Traceback|rror:|falls written|diagnostics written|Killed|OOM|RESOURCE_EXHAUSTED"

Q=$BIG/queue.txt; LOCK=$BIG/queue.lock; : > $Q
# name|ckpt|friction|steps|seeds|extra ; longest first
for p in 0 1 2 3 4; do echo "B_ice_S4_p$p|go1_broadwm|0.15|5000|${PARTS[$p]}|$S4" >> $Q; done
echo "B_ice_V0|go1_broadwm|0.15|5000|$ALL|$V0" >> $Q
echo "B_ice_Sq|go1_broadwm|0.15|5000|$ALL|$SQ" >> $Q
for p in 0 1 2 3 4; do echo "B_floor_S4_p$p|go1_broadwm|0.8|2000|${PARTS[$p]}|$S4" >> $Q; done
for p in 0 1 2 3 4; do echo "P_floor_S4_p$p|go1_paper|0.8|2000|${PARTS[$p]}|$S4" >> $Q; done
echo "B_floor_V0|go1_broadwm|0.8|2000|$ALL|$V0" >> $Q
echo "P_floor_V0|go1_paper|0.8|2000|$ALL|$V0" >> $Q
echo "B_floor_Sq|go1_broadwm|0.8|2000|$ALL|$SQ" >> $Q
echo "P_floor_Sq|go1_paper|0.8|2000|$ALL|$SQ" >> $Q

worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name ckpt fr steps seeds extra <<< "$job"
    echo "== $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt control.prior_only=false $extra \
      sim.friction=$fr sim.total_steps=$steps $COMMON "batch.seeds=[$seeds]" \
      output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz \
      output.model_name=$name output.condition=friction$fr --headless 2>&1 | grep -aE "$FILT" | head -6
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
echo "### stage28 start $(date)"
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage28 done $(date)"
