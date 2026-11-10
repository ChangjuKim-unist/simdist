#!/bin/bash
# Design doc 17 (docs/design_baseline_retrain_ko.md): SimDist baseline retrained from scratch on the broad-DR dataset
# (two seeds, paper recipe: scripts/train_model.py, batch 512), then on each model: SimDist planner (J, 15 conditions with
# traces), prior-only, calibrated heads from the traces, generator + q50, and two ablations.  Side group while the two
# trainings occupy GPUs 0/1: generator seeds 1/2, G0(d) (generator mean only, iterations=0) and the seed check on go1_broadwm.
# gpu1, 4 GPUs.  Usage:  nohup bash scripts/sim2sim/stage17_scratch.sh > results/stage17_scratch.log 2>&1 &
# Re-runnable: finished rows/files are skipped.  Queue jobs wait until their inputs (model, traces, head, generator) exist.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage17; TR=/nas_data/cjkim/simdist/traces_stage17; mkdir -p $OUT $TR
export SIMDIST_DIR=~/simdist
GPUS_LIST=${GPUS_LIST:-"0 1 2 3"}
DS=2026-11-05_11-39-08; DSDIR=datasets/sim/$DS
# HDF5 reads from the NAS (NFS) fail with EIO under 8 DataLoader workers (2026-11-10 11:34), so the dataset lives on the
# server's spare SSD (/mnt/newssd, mounted into the container by run.sh) and datasets/sim points there.
LOCALDS=/mnt/newssd/cjkim/simdist/datasets/sim/$DS
[ -f $LOCALDS/processed_data_go1dr_H-25_T-25/values.hdf5 ] || { mkdir -p $(dirname $LOCALDS); rsync -a /nas_data/cjkim/simdist/datasets/sim/$DS $(dirname $LOCALDS)/; }
ln -sfn $LOCALDS $DSDIR
echo "### stage17 start $(date) GPUs: $GPUS_LIST"

# ---- A. from-scratch trainings (paper recipe; the GPU is marked busy so the eval worker on it waits) ----
TRAIN_BATCH=${TRAIN_BATCH:-512}; TRAIN_STEPS=${TRAIN_STEPS:-150000}
train() { local gpu=$1 name=$2 seed=$3
  touch $OUT/gpu$gpu.busy
  if [ ! -f $OUT/train_$name.done ]; then
    echo "== train $name seed $seed batch $TRAIN_BATCH steps $TRAIN_STEPS on gpu$gpu $(date +%H:%M)"
    GPUS="\"device=$gpu\"" $RUN python scripts/train_model.py data.dataset_name=$DS system=go1dr run_name=$name \
      training.seed=$seed training.batch_size=$TRAIN_BATCH training.num_epochs=20 training.max_steps=$TRAIN_STEPS \
      training.warmup_steps=10000 training.decay_steps=$TRAIN_STEPS training.learning_rate=0.0002 training.end_learning_rate=0.0001 \
      training.eval_interval=10000 2>&1 | grep --line-buffered -E "Steps: [0-9]*000, |Error|error|Traceback|stop after|completed|OOM|memory" | cut -c1-240 > $OUT/train_$name.log
    if [ -d checkpoints/models/$name/$TRAIN_STEPS ]; then   # the trainer's "Steps:" lines do not reach the pipe; the final checkpoint is the proof
      touch $OUT/train_$name.done; echo "== train $name done $(date +%H:%M)"
    else echo "== train $name FAILED $(date +%H:%M) (see $OUT/train_$name.log)"; fi
  fi
  rm -f $OUT/gpu$gpu.busy
}

# ---- B. job queue ----
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
GEN="control.gen.enabled=true control.gen.weights=/workspace/simdist/results/stage15/gen_broad.npz"
gens() { echo "control.gen.enabled=true control.gen.weights=/workspace/simdist/results/stage17/gen_broad_s$1.npz"; }
PO="control.prior_only=true"
cal() { echo "control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage17/$1.npz control.calib.tau=0.5"; }
cal14() { echo "control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage14/$1.npz control.calib.tau=0.5"; }
run() { local gpu=$1 name=$2 model=$3 cond=$4; shift 4
  local csv=$OUT/eval_scratch_gpu$gpu.csv
  if cat $OUT/eval_scratch*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  local extra=""; [ "${1:-}" = trace ] && { extra="output.trace_npz=$TR/${name}_$cond.npz"; shift; }
  echo "== $name ($model) on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$model $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.terms_csv=/workspace/simdist/$OUT/terms_scratch_gpu$gpu.csv \
    output.model_name=$name output.condition=$cond $extra "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
genseed() { local gpu=$1 s=$2
  [ -f $OUT/gen_broad_s$s.npz ] && return
  echo "== generator seed $s on gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/gen_chunks.py --data $DSDIR/processed_data_go1dr_H-25_T-25 --out $OUT/gen_broad_s$s --seed $s 2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -6
}
head_train() { local gpu=$1 tag=$2 axis=$3 x
  case $axis in friction) x=F;; motor) x=M;; mass) x=A;; esac
  [ -f $OUT/head_${tag}_no$x.npz ] && return
  echo "== head $tag holdout $axis on gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/calib_head.py --traces "$TR/J_${tag}_*.npz" --holdout_axis $axis --out $OUT/head_${tag}_no$x 2>&1 | grep -v "WARNING\|cuStream\|^$" | tail -8
}
# a job is ready when its model has finished training and every stage17 file it names exists; HEAD needs the 15 traces
ready() { local j="$1" tok; set -- $j
  case $1 in
    GENSEED) return 0;;
    HEAD) [ $(ls $TR/J_${2}_*.npz 2>/dev/null | wc -l) -ge 15 ] && return 0 || return 1;;
  esac
  case $3 in go1_scratch_*) [ -f $OUT/train_$3.done ] || return 1;; esac
  for tok in "$@"; do case $tok in *results/stage17/*.npz) [ -f ${tok##*/workspace/simdist/} ] || return 1;; esac; done
  return 0
}
QUEUE=$OUT/queue_scratch.txt; LOCK=$OUT/queue_scratch.lock
COND15="nominal friction0.15 motor0.4 mass3 friction0.3 friction0.5 friction0.3_mass3 mass1 mass5 motor0.3 motor0.5 motor0.5_friction0.3 motor0.6 motor0.7_mass3 motor0.8"
{
  echo "GENSEED 1"; echo "GENSEED 2"
  echo "D_gen_mean go1_broadwm friction0.15 $GEN control.iterations=0"
  echo "D_gen_mean go1_broadwm motor0.4 $GEN control.iterations=0"
  for s in 1 2; do echo "E_gen_s${s}_q50 go1_broadwm friction0.15 $(gens $s) $(cal14 head_broadwm_noF)"; echo "E_gen_s${s}_q50 go1_broadwm motor0.4 $(gens $s) $(cal14 head_broadwm_noM)"; done
  for c in $COND15; do for s in 1 2; do echo "J_s$s go1_scratch_s$s $c trace"; done; done
  for c in nominal friction0.15 motor0.4 mass3; do for s in 1 2; do echo "P_s$s go1_scratch_s$s $c $PO"; done; done
  for s in 1 2; do for a in friction motor mass; do echo "HEAD s$s $a"; done; done
  for s in 1 2; do
    echo "GQ_s$s go1_scratch_s$s friction0.15 $GEN $(cal head_s${s}_noF)"
    echo "GQ_s$s go1_scratch_s$s motor0.4 $GEN $(cal head_s${s}_noM)"
    echo "GQ_s$s go1_scratch_s$s mass3 $GEN $(cal head_s${s}_noA)"
    echo "GQ_s$s go1_scratch_s$s nominal $GEN $(cal head_s${s}_noF)"
    echo "GJ_s$s go1_scratch_s$s friction0.15 $GEN"
    echo "Q_s$s go1_scratch_s$s friction0.15 $(cal head_s${s}_noF)"
  done
} > $QUEUE
echo "== queue: $(wc -l < $QUEUE) jobs"
pop() { ( flock 9; local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    if ready "$line"; then grep -v -x -F -- "$line" $QUEUE > $QUEUE.tmp; mv $QUEUE.tmp $QUEUE; echo "$line"; return; fi
  done < $QUEUE; echo "" ) 9>$LOCK; }
worker() { local gpu=$1 j idle=0
  while :; do
    while [ -f $OUT/gpu$gpu.busy ]; do sleep 120; done
    j=$(pop)
    if [ -z "$j" ]; then
      [ -s $QUEUE ] || break
      # nothing ready: give up only when no training and no other job can still unblock the queue
      if ! ls $OUT/gpu*.busy $OUT/running_gpu* >/dev/null 2>&1; then idle=$((idle+1)); [ $idle -ge 10 ] && { echo "== worker gpu$gpu: queue blocked, giving up $(date +%H:%M)"; break; }; else idle=0; fi
      sleep 120; continue
    fi
    idle=0; echo "$j" > $OUT/running_gpu$gpu
    set -- $j
    case $1 in GENSEED) genseed $gpu $2;; HEAD) head_train $gpu $2 $3;; *) run $gpu $j;; esac
    rm -f $OUT/running_gpu$gpu
  done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }

gpus=($GPUS_LIST)
train ${gpus[0]} go1_scratch_s1 1 &
sleep 90
train ${gpus[1]:-${gpus[0]}} go1_scratch_s2 2 &
sleep 30
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
for stem in ('eval_scratch', 'terms_scratch'):
    rows=[]
    for f in sorted(glob.glob(f'results/stage17/{stem}_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
    if rows:
        with open(f'results/stage17/{stem}.csv','w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print('merged', stem, len(rows), 'rows')
PY
echo "### stage17 done $(date)"
