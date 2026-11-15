#!/bin/bash
# Design doc 22 §13: conservative head refit (gpu1, 4 GPUs, ~1 h).
#   usage (gpu1): tmux new -d -s s22c 'bash scripts/sim2sim/stage22_cons.sh > results/stage22q_cons.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage22q; mkdir -p $OUT
B=go1_paper_very_weak_motor_dyn_b45000_s42; C=go1_paper_very_weak_motor_hrpred_q
DATA=sim2sim_very_weak_motor_go1_paper_b45000; HO=sim2sim_very_weak_motor_go1_paper_ho_b15000
P=go1_paper_very_weak_motor
ev() { local g=$1 name=$2 ckpt=$3; shift 3
  echo "== eval $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4 \
    sim.terrain=plane sim.total_steps=2000 sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' \
    output.csv=/workspace/simdist/$OUT/cons_eval_gpu$g.csv output.model_name=$name output.condition=motor0.4 "$@" --headless 2>&1 \
    | grep -E "Traceback|rror:" | head -4; }
ft() { local g=$1 name=$2; shift 2
  echo "== refit $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/finetune_model.py data.dataset_name=$DATA system=go1 checkpoint.resume_checkpoint=$B \
    run_name=$name training.max_steps=2000 training.eval_interval=2000 training.warmup_steps=200 training.seed=42 \
    loss=head_refit adapt.head_anchor=true "$@" 2>&1 | grep -E "Parameters to be optimized|Froze|Steps: 3739|Error|Traceback" | cut -c1-400
  echo "== refit $name done $(date +%H:%M)"; }
echo "### stage22 cons start $(date)"
( ft 0 ${P}_hrpred_cons1_q heads=reward_value loss.value_input=pred loss.weights.cons=1.0; ev 0 cons_pred_l1 ${P}_hrpred_cons1_q model.step=373920
  echo "== candidates $(date +%H:%M)"
  GPUS='"device=0"' $RUN python scripts/sim2sim/stage22_candidates.py --dataset $HO --sigmas 0.5 \
    --ckpts B:$B C:$C Cl1:${P}_hrpred_cons1_q Cl10:${P}_hrpred_cons10_q Dl10:${P}_hrenc_cons10_q Cv10:${P}_hrval_cons10_q \
    --out /workspace/simdist/$OUT/cons_candidates.json 2>&1 | grep -E "^sigma|Traceback|rror:" | cut -c1-4000 ) &
( ft 1 ${P}_hrpred_cons10_q heads=reward_value loss.value_input=pred loss.weights.cons=10.0; ev 1 cons_pred_l10 ${P}_hrpred_cons10_q model.step=373920 ) &
( ft 2 ${P}_hrenc_cons10_q heads=reward_value loss.value_input=enc loss.weights.cons=10.0; ev 2 cons_enc_l10 ${P}_hrenc_cons10_q model.step=373920 ) &
( ft 3 ${P}_hrval_cons10_q heads=value_only loss.value_input=pred loss.weights.cons=10.0; ev 3 cons_predval_l10 ${P}_hrval_cons10_q model.step=373920
  ev 3 simdist_adapted_c $B ) &
wait
echo "### stage22 cons done $(date)"
