#!/usr/bin/env bash
# Design doc 13 §8: local companion experiments to the gpu4 context chain (stage11_context_server.sh).
#   L1 visibility probe on the server dataset (CPU, in background)
#   L2 seed replicate go1_ctx_s2 (training.seed=43) + motor0.4 / nominal evals
#   L3 go1_ctx (from gpu4) + prior-first low-friction dynamics adaptation + friction0.15 / motor0.4 / nominal evals
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage12; mkdir -p $OUT
CSV=$OUT/eval_local.csv; CTX=$OUT/context_estimates_local.csv; CTXOPT=output.context_csv=/workspace/simdist/$CTX
SSH="ssh -o BatchMode=yes gpu4"
echo "### local-ctx start $(date)"

# ---- L1: server dataset -> local, visibility probe ----
until DS=$(timeout 60 $SSH 'cat ~/simdist/results/stage11/dataset.txt 2>/dev/null' | tail -1) && [ -n "$DS" ]; do sleep 300; done
echo "== dataset $DS $(date +%H:%M)"
mkdir -p datasets/sim/$DS
until timeout 10800 rsync -rlpt --exclude raw_data.hdf5 gpu4:simdist/datasets/sim/$DS/ datasets/sim/$DS/; do sleep 120; done
[ -d datasets/sim/$DS/processed_data_go1dr_H-25_T-25 ] || { echo "no processed data"; exit 1; }
echo "== probe $(date +%H:%M)"
( $RUN env JAX_PLATFORMS=cpu python scripts/sim2sim/visibility_probe.py --dataset $DS --system go1dr --num_windows 300000 --steps 6000 --out $OUT/probe_ctxdata.json > $OUT/probe_ctxdata.log 2>&1; echo "== probe done $(date +%H:%M)" ) &

# ---- evaluation helpers (same as the server chain) ----
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond ($*) $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -4
}

# ---- L2: seed replicate ----
train() { local name=$1; shift
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return 0
  echo "== train $name $(date +%H:%M)"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=$name \
    loss=world_model heads=all training.max_steps=60000 training.batch_size=128 training.warmup_steps=2000 training.decay_steps=60000 \
    training.end_learning_rate=0.00005 training.eval_interval=10000 "$@" 2>&1 | grep -E "Steps: |Error|Traceback|Added" | cut -c1-220 | tail -3
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 || { echo "train $name failed"; return 1; }
}
if train go1_ctx_s2 adapt.context.enabled=true training.seed=43; then
  for c in motor0.4 nominal; do run C_ctx_planner_s2 go1_ctx_s2 $c $CTXOPT; done
fi

# ---- L3: go1_ctx + low-friction prior-first dynamics adaptation ----
until timeout 60 $SSH 'grep -q "^== C_" ~/simdist/results/stage11_server.log' 2>/dev/null; do sleep 600; done   # training on gpu4 finished
until timeout 1800 rsync -rlpt gpu4:simdist/checkpoints/models/go1_ctx/ checkpoints/models/go1_ctx/; do sleep 120; done
ls -d checkpoints/models/go1_ctx/[0-9]* >/dev/null 2>&1 || { echo "no go1_ctx"; exit 1; }
echo "== go1_ctx copied $(date +%H:%M)"
DSF=sim2sim_friction0.15_go1_ctx
[ -d datasets/real/$DSF/raw ] || BASE=go1_ctx COND=friction0.15 DATASET=$DSF COLLECT_CKPT=go1_ctx ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
[ -d datasets/real/${DSF}_b45000/processed_data_go1_H-25_T-25 ] || BASE=go1_ctx COND=friction0.15 DATASET=$DSF BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
CK=go1_ctx_friction0.15_dyn_b45000_term
ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || BASE=go1_ctx RESUME_CKPT=go1_ctx COND=friction0.15 DATASET=$DSF TAG=term BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 $P finetune 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-120
ls -d checkpoints/models/$CK/[0-9]* >/dev/null 2>&1 || { echo "adapt failed"; exit 1; }
for c in friction0.15 motor0.4 nominal; do run C_ctx_fradapt_planner $CK $c $CTXOPT; done
run C_ctx_planner go1_ctx friction0.15 $CTXOPT   # un-adapted reference on the same machine (server also runs it)
wait
echo "### local-ctx done $(date)"
