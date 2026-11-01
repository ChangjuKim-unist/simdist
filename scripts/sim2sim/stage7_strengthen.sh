#!/usr/bin/env bash
# Design doc 11: systematic strengthening experiments S1-S6 (see docs/design_strengthen_ko.md).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage7; mkdir -p $OUT
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";; motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";; payload5) echo "sim.friction=0.8 sim.add_mass=5.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local csv=$1 name=$2 ckpt=$3 cond=$4; shift 4
  if [ -f $csv ] && grep -q "^$name,$cond," $csv; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}
collect() { local cond=$1 ds=$2 ckpt=$3 rounds=$4   # ROUNDS x 3 speeds x 5000 steps
  [ -d datasets/real/$ds/raw ] || COND=$cond DATASET=$ds COLLECT_CKPT=$ckpt ROUNDS=$rounds COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120; }
process() { local cond=$1 ds=$2 b=$3; [ -d datasets/real/${ds}_b$b/processed_data_go1_H-25_T-25 ] || COND=$cond DATASET=$ds BUDGETS=$b $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120; }
ident() { local ds=$1 out=$2; shift 2; $RUN python scripts/sim2sim/identify_physics.py dataset=$ds out=$out "$@" --headless 2>&1 | grep -E "MAP|error per|Error|Traceback" | cut -c1-200; }
echo "### stage7 start $(date)"
# ---- S1: planner value on the targeted head
CSV=$OUT/eval_s1.csv
for cond in motor0.4 nominal; do run $CSV H_targethead_alone go1_paper_targethead $cond control.prior_only=true; done
# ---- S2: identification accuracy curve (1-minute logs, two collecting policies)
for s in 0.3 0.6 0.8 0.5; do
  for spec in "go1_paper_broadhead:bh" "go1_paper:sd"; do IFS=: read ck tag <<< "$spec"
    ds=sim2sim_motor${s}_${tag}_1min; collect motor$s $ds $ck 1; process motor$s $ds 3000
    [ -f $OUT/id_s2_${tag}_s$s.json ] || { echo "== identify $tag true $s $(date +%H:%M)"; ident ${ds}_b3000 $OUT/id_s2_${tag}_s$s.json; }
  done
done
# ---- S6: identifier ablation on the 1-minute deployed-prior log at true 0.4
DS=sim2sim_very_weak_motor_priorfirst_b3000
[ -f $OUT/id_s6_noweight.json ] || { echo "== S6 no demand weight"; ident $DS $OUT/id_s6_noweight.json demand_weight=false; }
[ -f $OUT/id_s6_seg10.json ] || { echo "== S6 segment 10"; ident $DS $OUT/id_s6_seg10.json seg_len=10 skip=2; }
# ---- S3: matched-content control with the targeted head
CSV=$OUT/eval_s3.csv
for s in 0.5 0.4; do c=$([ $s = 0.4 ] && echo very_weak_motor || echo motor0.5); ds=sim2sim_${c}_targethead_onpolicy
  collect $c $ds go1_paper_targethead 3; process $c $ds 45000
  CK=go1_paper_targethead_${c}_dyn_b45000_s3
  ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || BASE=go1_paper_targethead COND=$c DATASET=$ds TAG=s3 BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 RESUME_CKPT=go1_paper_targethead $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
  run $CSV H_targethead_adapt_s$s $CK motor0.4
done
# ---- S4: second seed of the targeted head
EXP=$OUT/expert_target_s2.txt
if [ ! -f $EXP ]; then before=$(ls -t checkpoints/rl | head -1); echo "== expert 0.3-0.7 seed 2 $(date +%H:%M)"
  $RUN env GO1_MOTOR_DR=0.3,0.7 python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 --seed 2 2>&1 | grep -E "Traceback|rror:" | head -2
  after=$(ls -t checkpoints/rl | head -1); [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ] && echo "$after" > $EXP || echo "!! expert seed 2 failed"; fi
if [ -f $EXP ]; then RL=$(cat $EXP)
  [ -f checkpoints/rl/$RL/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $RL --headless 2>&1 | grep -E "Error|Traceback" | tail -1
  if [ ! -f $OUT/dataset_target_s2 ]; then before=$(ls datasets/sim | sort | tail -1); echo "== generate target s2 data $(date +%H:%M)"
    $RUN env GO1_MOTOR_DR=0.3,0.7 python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 2>&1 | grep -E "Error|Traceback" | head -2
    ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] && echo "$ds" > $OUT/dataset_target_s2; $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2; fi
  DS2=$(cat $OUT/dataset_target_s2)
  ls -d checkpoints/models/go1_paper_targethead_s2/[0-9]* >/dev/null 2>&1 || $RUN python scripts/finetune_model.py data.dataset_name=$DS2 system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_targethead_s2 loss=world_model heads=policy_only training.max_steps=60000 training.eval_interval=10000 2>&1 | grep -E "Error|Traceback" | tail -1
  CSV=$OUT/eval_s4.csv; for cond in motor0.4 nominal; do run $CSV H_targethead_s2_planner go1_paper_targethead_s2 $cond; done
fi
# ---- S5: payload axis
CSV=$OUT/eval_s5.csv
for spec in "go1_paper_broadhead:bh" "go1_paper:sd"; do IFS=: read ck tag <<< "$spec"; ds=sim2sim_payload5_${tag}_1min; collect payload5 $ds $ck 1; process payload5 $ds 3000
  [ -f $OUT/id_s5_$tag.json ] || { echo "== identify payload $tag $(date +%H:%M)"; ident ${ds}_b3000 $OUT/id_s5_$tag.json "grid.lo=0.8" "grid.hi=1.1" "grid.n=4" "mass_grid=[0.0,1.5,3.0,4.5,6.0,7.5]" "replicas=8" "rounds=60"; }
done
EXP=$OUT/expert_payload.txt
if [ ! -f $EXP ]; then before=$(ls -t checkpoints/rl | head -1); echo "== payload expert (mass DR 3-7, no motor DR) $(date +%H:%M)"
  $RUN env GO1_MOTOR_DR=1.0,1.0 GO1_MASS_DR=3.0,7.0 python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 2>&1 | grep -E "Traceback|rror:" | head -2
  after=$(ls -t checkpoints/rl | head -1); [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ] && echo "$after" > $EXP || echo "!! payload expert failed"; fi
if [ -f $EXP ]; then RL=$(cat $EXP)
  [ -f checkpoints/rl/$RL/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $RL --headless 2>&1 | grep -E "Error|Traceback" | tail -1
  if [ ! -f $OUT/dataset_payload ]; then before=$(ls datasets/sim | sort | tail -1); echo "== generate payload data $(date +%H:%M)"
    $RUN env GO1_MOTOR_DR=1.0,1.0 GO1_MASS_DR=3.0,7.0 python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 2>&1 | grep -E "Error|Traceback" | head -2
    ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] && echo "$ds" > $OUT/dataset_payload; $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2; fi
  DSP=$(cat $OUT/dataset_payload)
  ls -d checkpoints/models/go1_paper_payloadhead/[0-9]* >/dev/null 2>&1 || $RUN python scripts/finetune_model.py data.dataset_name=$DSP system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_payloadhead loss=world_model heads=policy_only training.max_steps=60000 training.eval_interval=10000 2>&1 | grep -E "Error|Traceback" | tail -1
  for cond in payload5 nominal; do run $CSV A_simdist_planner go1_paper $cond; run $CSV H_broadhead_planner go1_paper_broadhead $cond; run $CSV H_payloadhead_planner go1_paper_payloadhead $cond; done
fi
echo "### stage7 done $(date)"
