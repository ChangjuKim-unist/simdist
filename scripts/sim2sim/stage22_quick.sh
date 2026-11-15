#!/bin/bash
# Design doc 22 §11 quick check (gpu1, 4 GPUs): refit reward/value heads of the motor-adapted go1_paper on predicted
# latents (C) and on true latents (D); short closed loop on motor0.4 (B, C, D) and nominal (C); offline fall AUC.
#   usage (gpu1): tmux new -d -s s22 'bash scripts/sim2sim/stage22_quick.sh > results/stage22q_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage22q; mkdir -p $OUT
B=go1_paper_very_weak_motor_dyn_b45000_s42; C=go1_paper_very_weak_motor_hrpred_q; D=go1_paper_very_weak_motor_hrtrue_q
DATA=sim2sim_very_weak_motor_go1_paper_b45000; HO=sim2sim_very_weak_motor_go1_paper_ho_b15000; STEPS=2000
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; esac; }
ev() { local g=$1 name=$2 ckpt=$3 cond=$4
  echo "== eval $name $cond gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=$STEPS \
    sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv \
    output.model_name=$name output.condition=$cond --headless 2>&1 | grep -E "Traceback|rror:" | head -3; }
ft() { local g=$1 name=$2 inp=$3
  echo "== refit $name ($inp) gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/finetune_model.py data.dataset_name=$DATA system=go1 checkpoint.resume_checkpoint=$B \
    run_name=$name training.max_steps=2000 training.eval_interval=2000 training.warmup_steps=200 training.seed=42 \
    loss=head_refit heads=reward_value loss.value_input=$inp 2>&1 | grep -E "Parameters to be optimized|Steps: 2000|Error|Traceback" | cut -c1-250
  echo "== refit $name done $(date +%H:%M)"; }
echo "### stage22 quick start $(date)"
( ft 0 $C pred; ev 0 hr_pred $C motor0.4 ) &
( ft 1 $D true; ev 1 hr_true $D motor0.4 ) &
( ev 2 simdist_adapted $B motor0.4
  while [ ! -d checkpoints/models/$C ] || [ ! -d checkpoints/models/$D ] || ! ls -d checkpoints/models/$C/[0-9]* checkpoints/models/$D/[0-9]* >/dev/null 2>&1; do sleep 30; done
  sleep 60; echo "== offline $(date +%H:%M)"
  GPUS='"device=2"' $RUN python scripts/sim2sim/stage22_offline.py --dataset $HO --ckpts B:$B C:$C D:$D \
    --out /workspace/simdist/$OUT/offline_motor_ho.json 2>&1 | grep -E "windows|RESULT|Traceback|rror:" | cut -c1-2000 ) &
( while ! ls -d checkpoints/models/$C/[0-9]* >/dev/null 2>&1; do sleep 30; done; sleep 60; ev 3 hr_pred $C nominal ) &
wait
echo "### stage22 quick done $(date)"
