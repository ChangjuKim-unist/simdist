#!/bin/bash
# Design doc 29 gate G1 (docs/design29_gate_physcode_ko.md §3): robust-until-confident scoring over {adapted, pretrained}
# during the first 2 s after every reset, vs the adapted model alone, on nominal / ice / motor 40 % (design-20 protocol),
# plus the ramp-start check Q1r. usage (gpu1): tmux new -d -s s29 'bash scripts/sim2sim/stage29.sh > results/stage29_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage29; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
COMMON="sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9] batch.seeds=[11,12,13,14,15,16]"
phys() { case $1 in nom) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mot) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  ice) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
base() { [ $1 = P ] && echo go1_paper || echo go1_broadwm; }
ck() { local b; b=$(base $1); case $2 in PRE) echo $b;; ICE|ICER) echo ${b}_friction0.15_dyn_b45000_s42;; MOT|MOTR) echo ${b}_very_weak_motor_dyn_b45000_s42;; esac; }
FILT="Traceback|rror:|falls written|diagnostics written|robust scoring|Killed|OOM|RESOURCE_EXHAUSTED"
Q=$BIG/queue.txt; LOCK=$BIG/queue.lock; : > $Q
# name|model|arm|cond|ramp ; ice cells first (most resets -> most robust compute)
for c in ice mot nom; do for m in P B; do for a in MOTR ICER MOT ICE PRE; do echo "${m}_${a}_${c}|$m|$a|$c|0" >> $Q; done; done; done
for m in P B; do for a in PRE MOT; do echo "${m}_${a}_ice_ramp|$m|$a|ice|1" >> $Q; done; for a in PRE ICE; do echo "${m}_${a}_mot_ramp|$m|$a|mot|1" >> $Q; done; done

worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name m a c ramp <<< "$job"
    local extra=""
    case $a in ICER|MOTR) extra="control.robust.enabled=true control.robust.models=[$(base $m)]";; esac
    [ "$ramp" = 1 ] && extra="$extra sim.cmd_ramp_steps=50"
    echo "== $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$(ck $m $a) control.prior_only=false $extra \
      $(phys $c) $COMMON output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz \
      output.model_name=$name output.condition=$c --headless 2>&1 | grep -aE "$FILT" | head -6
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
echo "### stage29 start $(date) jobs $(wc -l < $Q)"
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage29 done $(date)"
