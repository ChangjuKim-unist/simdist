#!/bin/bash
# Design doc 38 (docs/design38_privileged_mask_ko.md): the paper RL expert with privileged observation groups replaced by their
# nominal mid-walk reference. Masks none/F/H/P/A x levels nominal/legR50/slope5/tow20/mot50 (25 runs). Local (one GPU).
# usage: bash scripts/sim2sim/stage38.sh > results/stage38.log 2>&1      SMOKE=1: 400 steps, seed 11, speeds 0.3/0.9
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=${BIG:-results/stage38}; GPU=${GPU:-0}; SMOKE=${SMOKE:-0}; mkdir -p $BIG
RL=2026-10-01_10-38-36; IT=4999; M=go1_paper
TOTAL=5000; SEEDS="[11,12,13,14,15,16]"; SPEEDS="[0.3,0.6,0.9]"
if [ $SMOKE = 1 ]; then TOTAL=400; SEEDS="[11]"; SPEEDS="[0.3,0.9]"; fi
phys() { [ $1 = nominal ] && return; BASE=$M COND=$1 bash -c "source <(sed -n '/^condition_overrides()/,/^}/p' $P); condition_overrides $1"; }
F="feet_body_forces"; H="feet_height"; PP="base_mass,material_properties,joint_stiffness,joint_damping,joint_friction"
declare -A MASK=([none]="" [F]="$F" [H]="$H" [P]="$PP" [A]="$F,$H,$PP")
REF=$BIG/mask_ref.npz
run() { local m=$1 c=$2 name=${2}_$1 extra=""
  [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; return; }
  if [ $m = none ] && [ $c = nominal ]; then extra="policy.mask_record=$REF"
  elif [ $m != none ]; then extra="policy.mask_terms=[${MASK[$m]}] policy.mask_ref=$REF"; fi
  echo "== $name $(date +%H:%M)"
  GPUS="\"device=$GPU\"" $RUN env SIMDIST_RL_DIR=/workspace/simdist/checkpoints/rl python scripts/simulate_batch.py model.checkpoint=$M \
    policy.rl_run=$RL policy.iteration=$IT $extra $(phys $c) sim.terrain=plane sim.episode_length_s=100 "batch.speeds=$SPEEDS" \
    "batch.seeds=$SEEDS" sim.total_steps=$TOTAL output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv \
    output.diag_npz=$BIG/diag_$name.npz output.model_name=expert_$m output.condition=$c --headless 2>&1 \
    | grep -aE "Traceback|rror:|design 38: masking|mask reference written|Killed|OOM" | grep -v cuStreamGetGreenCtx | head -6
  [ -f $BIG/eval_$name.csv ] || echo "!! $name failed"
  echo "-- $name done $(date +%H:%M)"; }
echo "### stage38 start $(date)"; git log --oneline -1 | cut -c1-60
run none nominal   # writes the nominal reference first
[ -f $REF ] || { echo "!! no mask reference"; exit 1; }
for c in nominal legR50 slope5 tow20 mot50; do for m in none F H P A; do run $m $c; done; done
echo "### stage38 done $(date)"
