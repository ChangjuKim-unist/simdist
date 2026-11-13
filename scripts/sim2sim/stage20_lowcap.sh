#!/bin/bash
# Design doc 20 (docs/design20_lowcap_ko.md): low-capacity latent-dynamics adaptation and forgetting, sim-to-sim on gpu1.
# Phases (each drained by one worker per GPU through a flock queue):
#   A collect  training logs for go1_broadwm (friction0.15, very_weak_motor; 3 rounds x 3 speeds x 5000 steps = 45k),
#              held-out logs for both models x 4 conditions (1 round, seed base 2000), appendix prior-only logs (go1_paper)
#   B process  aggregate + window every dataset (CPU)
#   C finetune 2 models x 2 orders x V1..V6 x 2 seeds, then the revisit chain friction -> motor (V1, V2, V6)
#   D policy-only evaluations (gate P3: identical to V0) and latent-error metric on the held-out logs
#   E planner evaluations, most important cells first
#   usage: GPUS_LIST="0 1 2 3" nohup bash scripts/sim2sim/stage20_lowcap.sh > results/stage20_server.log 2>&1 &
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage20; mkdir -p $OUT $OUT/pw
GPUS_LIST=${GPUS_LIST:-"0 1 2 3"}; PHASES=${PHASES:-"A B C D E"}
MODELS="go1_broadwm go1_paper"; CONDS="friction0.15 very_weak_motor nominal payload"; VARS="dyn lora1 lora4 affine last dyn_replay"; SEEDS="42 43"
simds() { [ "$1" = go1_broadwm ] && echo 2026-11-05_11-39-08 || echo 2026-10-02_01-18-59; }
replaysys() { [ "$1" = go1_broadwm ] && echo go1dr || echo ""; }
evalcond() { case $1 in friction0.15) echo friction0.15;; very_weak_motor) echo motor0.4;; nominal) echo nominal;; payload) echo mass3;; esac; }
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mass3) echo "sim.friction=0.8 sim.add_mass=3.0 sim.motor_strength=1.0";; esac; }
# training datasets: go1_paper logs already exist (design 11/14); go1_broadwm logs are collected here
trainds() { case "$1 $2" in "go1_paper friction0.15") echo sim2sim_friction0.15_go1_paper;; "go1_paper very_weak_motor") echo sim2sim_very_weak_motor_go1_paper;;
  *) echo sim2sim_${2}_${1};; esac; }
hods() { echo sim2sim_${2}_${1}_ho; }
ckname() { echo ${1}_${2}_${3}_b45000_s$4; }          # model cond variant seed
echo "### stage20 start $(date) phases=$PHASES gpus=$GPUS_LIST"

# ---------- job functions (first arg = gpu) ----------
collect_train() { local g=$1 m=$2 c=$3; local ds; ds=$(trainds $m $c)
  [ -d datasets/real/$ds/raw ] && { echo "== collect_train $ds exists"; return; }
  echo "== collect_train $ds gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m COND=$c DATASET=$ds COLLECT_CKPT=$m ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160; }
collect_ho() { local g=$1 m=$2 c=$3; local ds; ds=$(hods $m $c)
  [ -d datasets/real/$ds/raw ] && { echo "== collect_ho $ds exists"; return; }
  echo "== collect_ho $ds gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m COND=$c DATASET=$ds COLLECT_CKPT=$m ROUNDS=1 COLLECT_STEPS=5000 SEED_BASE=2000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160; }
collect_app() { local g=$1 c=$2; local ds=sim2sim_${c}_go1_paper_prior
  [ -d datasets/real/$ds/raw ] && { echo "== collect_app $ds exists"; return; }
  echo "== collect_app $ds gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=go1_paper COND=$c DATASET=$ds COLLECT_CKPT=go1_paper ROUNDS=1 COLLECT_STEPS=5000 SEED_BASE=3000 COLLECT_EXTRA=control.prior_only=true $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160; }
process_ds() { local g=$1 ds=$2 b=$3
  [ -d datasets/real/${ds}_b$b/processed_data_go1_H-25_T-25 ] && return
  echo "== process ${ds}_b$b gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" DATASET=$ds BUDGETS=$b $P process 2>&1 | grep -E "^==|Error|Traceback|Total" | cut -c1-160; }
finetune() { local g=$1 m=$2 c=$3 v=$4 s=$5; local ds; ds=$(trainds $m $c); local name; name=$(ckname $m $c $v $s)
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return
  echo "== finetune $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$m COND=$c DATASET=$ds TAG=s$s BUDGETS=45000 VARIANTS=$v FT_STEPS=3000 FT_EXTRA="training.seed=$s" \
    NOMINAL_DATASET=$(simds $m) REPLAY_SYSTEM=$(replaysys $m) $P finetune 2>&1 | grep -E "^==|optimized|Attached|Replay|Steps: |Error|Traceback" | cut -c1-200; }
finetune_revisit() { local g=$1 m=$2 v=$3 s=$4; local from; from=$(ckname $m friction0.15 $v $s); local name=${m}_very_weak_motor_${v}_b45000_fr2mo_s$s
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && return
  ls -d checkpoints/models/$from/[0-9]* >/dev/null 2>&1 || { echo "== revisit: missing $from"; return; }
  echo "== finetune_revisit $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$from COND=very_weak_motor DATASET=$(trainds $m very_weak_motor) TAG=fr2mo_s$s BUDGETS=45000 VARIANTS=$v FT_STEPS=3000 FT_EXTRA="training.seed=$s" \
    NOMINAL_DATASET=$(simds $m) REPLAY_SYSTEM=$(replaysys $m) $P finetune 2>&1 | grep -E "^==|optimized|Attached|Replay|Steps: |Error|Traceback" | cut -c1-200; }
evalrun() { local g=$1 arm=$2 name=$3 ckpt=$4 cond=$5; local csv=$OUT/eval_${arm}_gpu$g.csv; local extra=""
  [ $arm = prior ] && extra=control.prior_only=true
  if cat $OUT/eval_${arm}_gpu*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  [ -d checkpoints/models/$ckpt ] || { echo "== eval: missing $ckpt"; return; }
  echo "== eval_$arm $name on $cond gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.model_name=$name output.condition=$cond $extra --headless 2>&1 | grep -E "Traceback|rror:" | head -3; }
laterr() { local g=$1 tag=$2 ckpt=$3 m=$4; local dss=""; local c
  for c in $CONDS; do dss="$dss $(hods $m $c)_b15000"; done
  [ -d checkpoints/models/$ckpt ] || { echo "== laterr: missing $ckpt"; return; }
  echo "== laterr $tag $ckpt gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/latent_error.py --ckpt $ckpt --datasets $dss --out $OUT/latent_error_gpu$g.csv --tag $tag 2>&1 | grep -E "RESULT|skip|Traceback|rror:" | cut -c1-200; }
laterr_app() { local g=$1 ckpt=$2; local dss=""; local c
  for c in nominal friction0.15 very_weak_motor; do dss="$dss sim2sim_${c}_go1_paper_prior_b15000 $(hods go1_paper $c)_b15000"; done
  echo "== laterr_app $ckpt gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/latent_error.py --ckpt $ckpt --datasets $dss --out $OUT/latent_error_app_gpu$g.csv --tag app \
    --per_window "/workspace/simdist/$OUT/pw/pw_{ckpt}_{dataset}.npz" 2>&1 | grep -E "RESULT|skip|Traceback|rror:" | cut -c1-200; }

# ---------- queue ----------
QUEUE=$OUT/queue.txt; LOCK=$OUT/queue.lock
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
runphase() { local name=$1; echo "== phase $name: $(wc -l < $QUEUE) jobs $(date +%H:%M)"; local g; for g in $GPUS_LIST; do worker $g & sleep 15; done; wait; echo "== phase $name done $(date +%H:%M)"; }
# worker calls "$j" = "fn args" with gpu inserted after fn
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; set -- $j; local fn=$1; shift; $fn $gpu "$@"; done; }

if [[ " $PHASES " == *" A "* ]]; then
  : > $QUEUE
  for c in friction0.15 very_weak_motor; do echo "collect_train go1_broadwm $c" >> $QUEUE; done
  for c in nominal friction0.15 very_weak_motor; do echo "collect_app $c" >> $QUEUE; done
  for m in $MODELS; do for c in $CONDS; do echo "collect_ho $m $c" >> $QUEUE; done; done
  runphase A
fi
if [[ " $PHASES " == *" B "* ]]; then
  : > $QUEUE
  for c in friction0.15 very_weak_motor; do echo "process_ds $(trainds go1_broadwm $c) 45000" >> $QUEUE; done
  for m in $MODELS; do for c in $CONDS; do echo "process_ds $(hods $m $c) 15000" >> $QUEUE; done; done
  for c in nominal friction0.15 very_weak_motor; do echo "process_ds sim2sim_${c}_go1_paper_prior 15000" >> $QUEUE; done
  runphase B
fi
if [[ " $PHASES " == *" C "* ]]; then
  : > $QUEUE
  for s in $SEEDS; do for m in $MODELS; do for c in friction0.15 very_weak_motor; do for v in $VARS; do echo "finetune $m $c $v $s" >> $QUEUE; done; done; done; done
  runphase C1
  : > $QUEUE
  for s in $SEEDS; do for m in $MODELS; do for v in dyn lora1 dyn_replay; do echo "finetune_revisit $m $v $s" >> $QUEUE; done; done; done
  runphase C2
fi
# checkpoint list: tag name ckpt model
allckpts() { local m c v s
  for m in $MODELS; do echo "V0 ${m}_V0 $m $m"; done
  for s in $SEEDS; do for m in $MODELS; do for c in friction0.15 very_weak_motor; do for v in $VARS; do echo "$v ${m}_${c}_${v}_s$s $(ckname $m $c $v $s) $m"; done; done; done; done
  for s in $SEEDS; do for m in $MODELS; do for v in dyn lora1 dyn_replay; do echo "revisit ${m}_fr2mo_${v}_s$s ${m}_very_weak_motor_${v}_b45000_fr2mo_s$s $m"; done; done; done; }
if [[ " $PHASES " == *" D "* ]]; then
  : > $QUEUE
  allckpts | while read tag name ckpt m; do for c in $CONDS; do echo "evalrun prior $name $ckpt $(evalcond $c)" >> $QUEUE; done; done
  allckpts | while read tag name ckpt m; do echo "laterr $tag $ckpt $m" >> $QUEUE; done
  for ck in go1_paper $(ckname go1_paper friction0.15 dyn 42) $(ckname go1_paper very_weak_motor dyn 42) $(ckname go1_paper friction0.15 lora1 42) $(ckname go1_paper very_weak_motor lora1 42); do echo "laterr_app $ck" >> $QUEUE; done
  runphase D
  python3 - <<'PY'
import csv, glob, collections
rows = []
for f in glob.glob('results/stage20/eval_prior_gpu*.csv'): rows += list(csv.DictReader(open(f)))
by = collections.defaultdict(dict)
for r in rows: by[(r['condition'], float(r['speed']), int(r['seed']))][r['model']] = r
bad = 0
for k, d in by.items():
    for m in ('go1_broadwm', 'go1_paper'):
        ref = d.get(f'{m}_V0')
        if ref is None: continue
        for name, r in d.items():
            if not name.startswith(m) or name == f'{m}_V0': continue
            if r['num_falls'] != ref['num_falls'] or abs(float(r['run_reward_per_step']) - float(ref['run_reward_per_step'])) > 1e-3:
                bad += 1; print('P3 differs:', k, name, r['num_falls'], ref['num_falls'], r['run_reward_per_step'], ref['run_reward_per_step'])
print('P3 gate: policy-only rows differing from V0 =', bad, 'of', sum(len(d) for d in by.values()))
PY
fi
if [[ " $PHASES " == *" E "* ]]; then
  : > $QUEUE
  pr() { echo "evalrun planner $1 $2 $3" >> $QUEUE; }
  main="friction0.15 motor0.4 nominal"
  for m in $MODELS; do for c in $main; do pr ${m}_V0 $m $c; done; done
  for m in $MODELS; do for v in dyn lora1 dyn_replay; do for c in $main; do pr ${m}_friction0.15_${v}_s42 $(ckname $m friction0.15 $v 42) $c; done; done; done
  for m in $MODELS; do for v in dyn lora1 dyn_replay; do for c in $main; do pr ${m}_very_weak_motor_${v}_s42 $(ckname $m very_weak_motor $v 42) $c; done; done; done
  for m in $MODELS; do for cc in friction0.15 very_weak_motor; do for v in lora4 affine last; do for c in $main; do pr ${m}_${cc}_${v}_s42 $(ckname $m $cc $v 42) $c; done; done; done; done
  for m in $MODELS; do for cc in friction0.15 very_weak_motor; do for v in $VARS; do for c in $main; do pr ${m}_${cc}_${v}_s43 $(ckname $m $cc $v 43) $c; done; done; done; done
  for m in $MODELS; do pr ${m}_V0 $m mass3; for s in $SEEDS; do for cc in friction0.15 very_weak_motor; do for v in $VARS; do pr ${m}_${cc}_${v}_s$s $(ckname $m $cc $v $s) mass3; done; done; done; done
  for s in $SEEDS; do for m in $MODELS; do for v in dyn lora1 dyn_replay; do for c in friction0.15 motor0.4; do pr ${m}_fr2mo_${v}_s$s ${m}_very_weak_motor_${v}_b45000_fr2mo_s$s $c; done; done; done; done
  runphase E
fi
python3 - <<'PY'
import csv, glob
for arm in ('prior', 'planner'):
    rows = []
    for f in sorted(glob.glob(f'results/stage20/eval_{arm}_gpu*.csv')): rows += list(csv.DictReader(open(f)))
    if rows:
        with open(f'results/stage20/eval_{arm}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(arm, 'merged', len(rows), 'rows')
rows = []
for f in sorted(glob.glob('results/stage20/latent_error_gpu*.csv')): rows += list(csv.DictReader(open(f)))
if rows:
    with open('results/stage20/latent_error.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('latent_error merged', len(rows))
PY
echo "### stage20 done $(date)"
