#!/bin/bash
# Design doc 33 (docs/design33_two_model_ko.md): two models without weight sharing + policy start + online one-step error
# switch (system T), against the single replay model + policy start (system R), three training seeds from the start.
#   F1 fine-tunes S   = pretrained + simulation planner windows (replay: pretraining windows), seeds 42 43 44, 2 models
#   F2 fine-tunes A   = S(seed s) + adaptation data, dynamics only (4 cells x 3 seeds)
#                 D   = dynamics-only fine-tune of the pretrained model, seeds 43 44 (seed 42 = design 29)
#                 R   = no token + pretraining-window replay, seeds 43 44 (seed 42 = design 32, tag d32s42)
#   E  closed loop (design-29 protocol: 18 robots x 5000 steps), names {P|B}_POL_c, {P|B}_PREps_c, {P|B}_{ICE|MOT}{arm}_{c}
#      arm = cal{s} (calibration of theta, robots 21-26, 2000 steps, own condition), T{s}, R{s}, d{s}
# usage (gpu1): tmux new -d -s s33 'bash scripts/sim2sim/stage33.sh > results/stage33_server.log 2>&1'
# SMOKE=1: tiny fine-tunes and four 150-step closed loops on the paper model (local check, not a result)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=${BIG:-/mnt/newssd/cjkim/simdist/results/stage33}; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; SMOKE=${SMOKE:-0}
FT_STEPS=3000; TOTAL=5000; CAL_TOTAL=2000; PS=125; WIN=50; QTL=95; MODELS="go1_paper go1_broadwm"; CONDS="friction0.15 very_weak_motor"; SEEDS="42 43 44"
[ $SMOKE = 1 ] && { FT_STEPS=30; TOTAL=150; CAL_TOTAL=150; PS=30; WIN=20; QTL=50; MODELS=go1_paper; CONDS=friction0.15; SEEDS=42; }
simds() { [ -n "${SMOKE_SIMDS:-}" ] && { echo $SMOKE_SIMDS; return; }; [ "$1" = go1_broadwm ] && echo 2026-11-05_11-39-08 || echo paper_sub8192; }
replaysys() { [ "$1" = go1_broadwm ] && echo go1dr || echo ""; }
t() { [ $SMOKE = 1 ] && echo ${1}smk || echo $1; }                   # checkpoint tag (smoke checkpoints get their own names)
ckS() { echo ${1}_nominal_dyn_replay_b45000_$(t S33s$2); }           # model, seed
ckA() { echo ${1}_${2}_dyn_b45000_$(t fS33s$3); }                    # model, condition, seed
ckD() { echo ${1}_${2}_dyn_b45000_$(t s$3); }
ckR() { echo ${1}_${2}_dyn_replay_b45000_$(t d32s$3); }
echo "### stage33 start $(date)"

# ---- phase F: fine-tunes (job queue: checkpoint name|model|resume from|condition|dataset|variant|seed|tag) -----------
FQ=$BIG/ft_queue.txt; LOCK=$BIG/queue.lock
ft_worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $FQ; sed -i 1d $FQ")
    [ -z "$job" ] && break
    IFS='|' read -r name m resume c ds v s tag <<< "$job"
    ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && { echo "== $name exists"; continue; }
    echo "== finetune $name (from $resume) gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$resume COND=$c DATASET=$ds TAG=$tag BUDGETS=45000 VARIANTS=$v FT_STEPS=$FT_STEPS FT_EXTRA="training.seed=$s" \
      NOMINAL_DATASET=$(simds $m) REPLAY_SYSTEM=$(replaysys $m) $P finetune 2>&1 | grep -aE "^==|optimized|Replay|Steps: |Error|Traceback" | cut -c1-260
    ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 || echo "!! $name missing after the fine-tune"
    echo "-- $name done $(date +%H:%M)"
  done; }
: > $FQ
for m in $MODELS; do for s in $SEEDS; do echo "$(ckS $m $s)|$m|$m|nominal|sim2sim_nominal_${m}|dyn_replay|$s|$(t S33s$s)" >> $FQ; done; done
for g in $GPU_LIST; do ft_worker $g & sleep 20; done; wait
echo "== phase F1 done $(date +%H:%M)"
: > $FQ
for s in $SEEDS; do for m in $MODELS; do for c in $CONDS; do
  echo "$(ckA $m $c $s)|$m|$(ckS $m $s)|$c|sim2sim_${c}_${m}|dyn|$s|$(t fS33s$s)" >> $FQ
  echo "$(ckR $m $c $s)|$m|$m|$c|sim2sim_${c}_${m}|dyn_replay|$s|$(t d32s$s)" >> $FQ
  [ $SMOKE = 1 ] || echo "$(ckD $m $c $s)|$m|$m|$c|sim2sim_${c}_${m}|dyn|$s|s$s" >> $FQ
done; done; done
for g in $GPU_LIST; do ft_worker $g & sleep 20; done; wait
echo "== phase F2 done $(date +%H:%M)"

# ---- phase E: closed loop -------------------------------------------------------------------------------------------
COMMON="sim.terrain=plane sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9]"
phys() { case $1 in nom) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mot) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  ice) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
base() { [ $1 = P ] && echo go1_paper || echo go1_broadwm; }
cname() { [ $1 = ICE ] && echo friction0.15 || echo very_weak_motor; }
own() { [ $1 = ICE ] && echo ice || echo mot; }; oth() { [ $1 = ICE ] && echo mot || echo ice; }
FILT="Traceback|rror:|falls written|diagnostics written|robust scoring|design 33|Killed|OOM|RESOURCE_EXHAUSTED"
Q=$BIG/queue.txt; : > $Q
AD="ICE MOT"; MS="P B"; [ $SMOKE = 1 ] && { AD=ICE; MS=P; }
# queue line: run name|kind|model|adaptation|seed|condition
ref() { local kind=$1; shift; for c in "$@"; do for m in $MS; do echo "${m}_${kind}_${c}|$kind|$m|-|0|$c" >> $Q; done; done; }
cal() { local s=$1; for a in $AD; do for m in $MS; do echo "${m}_${a}cal${s}|cal|$m|$a|$s|$(own $a)" >> $Q; done; done; }
add() { local kind=$1 arm=$2 s=$3 which=$4; for a in $AD; do for m in $MS; do local c; case $which in own) c=$(own $a);; oth) c=$(oth $a);; nom) c=nom;; esac
  echo "${m}_${a}${arm}${s}_${c}|$kind|$m|$a|$s|$c" >> $Q; done; done; }
if [ $SMOKE = 1 ]; then cal 42; ref POL ice; add T T 42 own; add R R 42 own
else
  cal 42; ref POL ice mot; ref PREps ice mot                 # threshold of seed 42, policy alone, pretrained + policy start
  add T T 42 own; add T T 42 oth; add R R 42 own; add R R 42 oth
  cal 43; cal 44
  add T T 43 own; add T T 43 oth; add R R 43 own; add R R 43 oth
  add T T 44 own; add T T 44 oth; add R R 44 own; add R R 44 oth
  add D d 43 own; add D d 44 own; add D d 43 oth; add D d 44 oth   # reference: dynamics-only fine-tune, seeds 43 44
  add T T 42 nom; add R R 42 nom
fi
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name kind m a s c <<< "$job"
    [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; continue; }
    local b; b=$(base $m); local ck extra="" seeds="[11,12,13,14,15,16]" total=$TOTAL
    local th=$BIG/theta_${m}_${a}_s${s}.txt
    case $kind in
      POL)   ck=$b; extra="control.prior_only=true";;
      PREps) ck=$b; extra="control.prior_only=false control.policy_start_steps=$PS";;
      cal)   ck=$(ckA $b $(cname $a) $s); extra="control.prior_only=false control.policy_start_steps=$PS control.err_switch.window=$WIN"; seeds="[21,22,23,24,25,26]"; total=$CAL_TOTAL;;
      R)     ck=$(ckR $b $(cname $a) $s); extra="control.prior_only=false control.policy_start_steps=$PS";;
      D)     ck=$(ckD $b $(cname $a) $s); extra="control.prior_only=false";;
      T)     ck=$(ckA $b $(cname $a) $s)
             local w=0; while [ ! -s $th ] && [ $w -lt 120 ]; do sleep 60; w=$((w + 1)); done
             [ -s $th ] || { echo "!! $name skipped: no threshold file $th"; continue; }
             extra="control.prior_only=false control.policy_start_steps=$PS control.robust.enabled=true control.robust.models=[$(ckS $b $s)] control.robust.mode=select"
             extra="$extra control.err_switch.enabled=true control.err_switch.theta=$(cat $th) control.err_switch.window=$WIN control.err_switch.fallback=model";;
    esac
    echo "== $name gpu$g $ck $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ck $extra \
      $(phys $c) $COMMON sim.total_steps=$total batch.seeds=$seeds output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz \
      output.model_name=$name output.condition=$c --headless 2>&1 | grep -aE "$FILT" | head -8
    if [ $kind = cal ]; then
      GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage33_calib.py --diag $BIG/diag_$name.npz --out $th --q $QTL 2>&1 | grep -aE "^theta|rror" | head -3
    fi
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage33 done $(date)"
