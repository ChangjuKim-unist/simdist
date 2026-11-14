#!/bin/bash
# Design doc 21 gate 0 (docs/design21_gate0_ctxopt_ko.md): fitted context slot vs inferred / oracle / full dynamics FT
# on go1_ctx, friction0.15 adaptation data collected by go1_broadwm. 12 planner runs on 4 GPUs.
#   usage (gpu1): tmux new -d -s s21 'bash scripts/sim2sim/stage21_gate0.sh > results/stage21_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage21; mkdir -p $OUT
DATA=sim2sim_friction0.15_go1_broadwm
FT=go1_ctx_friction0.15_dyn_b45000_bw
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
ev() { local g=$1 name=$2 ckpt=$3 cond=$4; shift 4
  echo "== eval $name $cond gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 \
    sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$OUT/eval_gpu$g.csv \
    output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:|override|Oracle" | head -3; }
echo "### stage21 gate0 start $(date)"

# GPU0: fit xi (frozen go1_ctx), then the fitted-xi arm on all three conditions
(
  GPUS='"device=0"' $RUN python scripts/sim2sim/stage21_ctx_opt.py --ckpt go1_ctx --dataset ${DATA}_b45000 \
    --true_xi 1.0 0.15 0.0 --out /workspace/simdist/$OUT/xi_friction0.15.json 2>&1 | grep -E "windows|held-out|^step (1|500|1000) |RESULT|Traceback|rror:" | cut -c1-300
  XI=$(python3 -c "import json;print(','.join(f'{v:.5f}' for v in json.load(open('$OUT/xi_friction0.15.json'))['xi_fit_std']))")
  echo "== fitted xi (standardized) [$XI]"
  for c in friction0.15 motor0.4 nominal; do ev 0 go1_ctx_xifit go1_ctx $c "sim.context_override_xi=[$XI]"; done
) &
# GPU1: inferred psi (design 13 model as is)
( for c in friction0.15 motor0.4 nominal; do ev 1 go1_ctx go1_ctx $c; done ) &
# GPU2: oracle psi[:3] = true physics of each condition
( for c in friction0.15 motor0.4 nominal; do ev 2 go1_ctx_oracle go1_ctx $c sim.context_oracle=true; done ) &
# GPU3: full dynamics fine-tune on the same data (design 20 V1 recipe), then evaluate with inferred psi
(
  echo "== finetune $FT gpu3 $(date +%H:%M)"
  GPUS='"device=3"' BASE=go1_ctx RESUME_CKPT=go1_ctx COND=friction0.15 DATASET=$DATA TAG=bw BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 \
    FT_EXTRA="training.seed=42" $P finetune 2>&1 | grep -E "^==|optimized|Steps: 3000|Error|Traceback" | cut -c1-200
  for c in friction0.15 motor0.4 nominal; do ev 3 go1_ctx_fullft $FT $c; done
) &
wait
echo "### stage21 gate0 done $(date)"
