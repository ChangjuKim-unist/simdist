#!/bin/bash
# Design doc 22 §12: why did the predicted-latent head refit collapse? (gpu1, 4 GPUs, ~40 min)
#   usage (gpu1): tmux new -d -s s22d 'bash scripts/sim2sim/stage22_diag.sh > results/stage22q_diag.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage22q; mkdir -p $OUT
B=go1_paper_very_weak_motor_dyn_b45000_s42; C=go1_paper_very_weak_motor_hrpred_q; D=go1_paper_very_weak_motor_hrtrue_q
HO=sim2sim_very_weak_motor_go1_paper_ho_b15000
ev() { local g=$1 name=$2 ckpt=$3; shift 3
  echo "== eval $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4 \
    sim.terrain=plane sim.total_steps=2000 sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' \
    output.csv=/workspace/simdist/$OUT/diag_eval_gpu$g.csv output.model_name=$name output.condition=motor0.4 "$@" --headless 2>&1 \
    | grep -E "Traceback|rror:|head taken|trace written" | head -4; }
echo "### stage22 diag start $(date)"
( ev 0 value_from_C $B model.heads_from.value=$C
  echo "== candidates $(date +%H:%M)"
  GPUS='"device=0"' $RUN python scripts/sim2sim/stage22_candidates.py --dataset $HO --ckpts B:$B C:$C D:$D \
    --out /workspace/simdist/$OUT/diag_candidates.json 2>&1 | grep -E "^sigma|Traceback|rror:" | cut -c1-3000 ) &
( ev 1 reward_from_C $B model.heads_from.reward=$C ) &
( ev 2 simdist_adapted_trace $B output.trace_npz=/workspace/simdist/$OUT/diag_trace_B.npz ) &
( ev 3 hr_pred_trace $C output.trace_npz=/workspace/simdist/$OUT/diag_trace_C.npz ) &
wait
echo "### stage22 diag done $(date)"
