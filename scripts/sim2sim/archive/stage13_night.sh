#!/usr/bin/env bash
# Design doc 14 stage 0 night queue on gpu4 (registered 2026-11-06 15:30, docs/design_trust_ko.md §1 "야간 서버 계획").
# Waits for stage13_trust0 to finish, then: (i) S0-2 probe on CPU; (ii) two go1_broadwm seed re-trainings on GPUs 0-1
# (ensemble baseline); (iii) a shared evaluation queue (adapted-model pairs, motor switch, planner budget sweep, physics
# grid = regime map and self-rollout data) served by workers on GPUs 2-3, joined by GPUs 0-1 when their trainings end.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage13; TR=/data1/cjkim/stage13; mkdir -p $OUT $TR
export SIMDIST_DIR=~/simdist
until grep -q "### trust0 done" results/stage13_server.log 2>/dev/null; do sleep 60; done
echo "### night start $(date)"

# ---- (i) S0-2 probe on the stage-0 traces (CPU inside the container) ----
$RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/trust_probe.py --traces "$TR/T_*.npz" --holdout mass3 --out /workspace/simdist/$OUT/probe_server.json 2>&1 | grep -v cuStream | grep -E "windows|hist|err|both|Traceback|rror" > $OUT/probe_server.log &
PROBE=$!

# ---- (ii) ensemble seeds (same recipe as stage11, seeds 43/44) ----
DS=$(cat results/stage11/dataset.txt)
train() { local gpu=$1 name=$2; shift 2
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return 0
  echo "== train $name on gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=$name \
    loss=world_model heads=all training.max_steps=60000 training.batch_size=128 training.warmup_steps=2000 training.decay_steps=60000 \
    training.end_learning_rate=0.00005 training.eval_interval=10000 "$@" 2>&1 | grep -E "Steps: |Error|Traceback" | cut -c1-200 | tail -2
  echo "== train $name done $(date +%H:%M)"
}

# ---- (iii) shared evaluation queue ----
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
SEEDS="[11,12,13,14,15,16]"
run() { local gpu=$1 name=$2 ckpt=$3 cond=$4; shift 4
  local csv=$OUT/eval_night_gpu$gpu.csv
  if [ -f $TR/${name}_${cond}.npz ] && cat $OUT/eval_night*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  echo "== $name on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond \
    output.trace_npz=$TR/${name}_${cond}.npz "$@" --headless 2>&1 | grep -E "Traceback|rror:|trace written|switched"
}
PO=control.prior_only=true; SW="sim.motor_switch_step=2500 sim.motor_switch_strength=0.8"
QUEUE=$OUT/night_queue.txt; LOCK=$OUT/night_queue.lock
cat > $QUEUE <<Q
A_ctxadapt_prior go1_ctx_friction0.15_dyn_b45000_term friction0.15 $PO
A_ctxadapt_planner go1_ctx_friction0.15_dyn_b45000_term friction0.15
A_ctxadapt_prior go1_ctx_friction0.15_dyn_b45000_term motor0.4 $PO
A_ctxadapt_planner go1_ctx_friction0.15_dyn_b45000_term motor0.4
A_paperfr_prior go1_paper_friction0.15_dyn_b45000_term friction0.15 $PO
A_paperfr_planner go1_paper_friction0.15_dyn_b45000_term friction0.15
A_papermotor_prior go1_paper_very_weak_motor_dyn_b45000_term motor0.4 $PO
A_papermotor_planner go1_paper_very_weak_motor_dyn_b45000_term motor0.4
S_ctx_prior_switch go1_ctx motor0.4 $PO $SW
S_ctx_planner_switch go1_ctx motor0.4 $SW
S_broadwm_prior_switch go1_broadwm motor0.4 $PO $SW
S_broadwm_planner_switch go1_broadwm motor0.4 $SW
B_broadwm_n100i3 go1_broadwm friction0.15 control.num_samples=100 control.iterations=3
B_broadwm_n1000i16 go1_broadwm friction0.15 control.num_samples=1000 control.iterations=16
B_broadwm_n100i3 go1_broadwm motor0.4 control.num_samples=100 control.iterations=3
B_broadwm_n1000i16 go1_broadwm motor0.4 control.num_samples=1000 control.iterations=16
G_broadwm_prior go1_broadwm motor0.3 $PO
G_broadwm_planner go1_broadwm motor0.3
G_broadwm_prior go1_broadwm friction0.3 $PO
G_broadwm_planner go1_broadwm friction0.3
G_broadwm_prior go1_broadwm mass5 $PO
G_broadwm_planner go1_broadwm mass5
G_broadwm_prior go1_broadwm motor0.6 $PO
G_broadwm_planner go1_broadwm motor0.6
G_broadwm_prior go1_broadwm friction0.5 $PO
G_broadwm_planner go1_broadwm friction0.5
G_broadwm_prior go1_broadwm motor0.5_friction0.3 $PO
G_broadwm_planner go1_broadwm motor0.5_friction0.3
G_broadwm_prior go1_broadwm motor0.7_mass3 $PO
G_broadwm_planner go1_broadwm motor0.7_mass3
G_broadwm_prior go1_broadwm friction0.3_mass3 $PO
G_broadwm_planner go1_broadwm friction0.3_mass3
G_broadwm_prior go1_broadwm motor0.8 $PO
G_broadwm_planner go1_broadwm motor0.8
G_broadwm_prior go1_broadwm mass1 $PO
G_broadwm_planner go1_broadwm mass1
A_paperfr_prior go1_paper_friction0.15_dyn_b45000_term motor0.4 $PO
A_paperfr_planner go1_paper_friction0.15_dyn_b45000_term motor0.4
A_papermotor_prior go1_paper_very_weak_motor_dyn_b45000_term friction0.15 $PO
A_papermotor_planner go1_paper_very_weak_motor_dyn_b45000_term friction0.15
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
worker 2 & sleep 20; worker 3 &
( train 0 go1_broadwm_s2 training.seed=43; worker 0 ) &
sleep 60
( train 1 go1_broadwm_s3 training.seed=44; worker 1 ) &
wait $PROBE; echo "== probe done $(date +%H:%M)"
wait
python3 - <<'PY'
import csv, glob
rows=[]
for f in sorted(glob.glob('results/stage13/eval_night_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
if rows:
    with open('results/stage13/eval_night.csv','w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('merged', len(rows), 'rows')
PY
echo "### night done $(date)"
