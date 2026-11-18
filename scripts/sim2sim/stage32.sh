#!/bin/bash
# Design doc 32 part A (docs/design32_recipe_ko.md): does the design-31 pass replicate with new training seeds, and which
# ingredient of the tok recipe keeps gain + retention?
#   F  fine-tunes (SimDist budget, 3000 steps), 2 models x 2 adaptation conditions:
#        tok          seeds 43, 44   token + pretraining windows + simulation planner windows (design 31 recipe)
#        dyn_replay2  seed 42        no token, both replay sets                                 (arm n2)
#        tok_noplan   seed 42        token, pretraining windows only                            (arm t1)
#        dyn_replay   seed 42        no token, pretraining windows only (design 20 recipe, same replay data as tok; tag d32s42) (arm n1)
#   E  closed loop (design-29 protocol: 18 robots x 5000 steps), run names {P|B}_{ICE|MOT}{arm}_{ice|mot|nom},
#      arm = t43A t43S t44A t44S (token model seed 43/44 with the adapted / simulation token), n2, t1A, n1
# usage (gpu1): tmux new -d -s s32 'bash scripts/sim2sim/stage32.sh > results/stage32_server.log 2>&1'
# SMOKE=1: one tiny fine-tune per variant and two 150-step closed loops (local check, not a result)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=${BIG:-/mnt/newssd/cjkim/simdist/results/stage32}; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; SMOKE=${SMOKE:-0}
FT_STEPS=3000; TOTAL=5000; MODELS="go1_paper go1_broadwm"; CONDS="friction0.15 very_weak_motor"
[ $SMOKE = 1 ] && { FT_STEPS=30; TOTAL=150; MODELS=${SMOKE_MODELS:-go1_paper}; CONDS=friction0.15; }
simds() { [ -n "${SMOKE_SIMDS:-}" ] && { echo $SMOKE_SIMDS; return; }; [ "$1" = go1_broadwm ] && echo 2026-11-05_11-39-08 || echo paper_sub8192; }
replaysys() { [ "$1" = go1_broadwm ] && echo go1dr || echo ""; }
echo "### stage32 start $(date)"

# ---- phase F: fine-tunes (job queue: model|condition|variant|seed|tag) ---------------------------------------------
FQ=$BIG/ft_queue.txt; LOCK=$BIG/queue.lock; : > $FQ
for m in $MODELS; do for c in $CONDS; do
  echo "$m|$c|dyn_replay2|42|s42" >> $FQ
  echo "$m|$c|tok|43|s43" >> $FQ; echo "$m|$c|tok|44|s44" >> $FQ
  echo "$m|$c|dyn_replay|42|d32s42" >> $FQ; echo "$m|$c|tok_noplan|42|s42" >> $FQ
done; done
ft_worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $FQ; sed -i 1d $FQ")
    [ -z "$job" ] && break
    IFS='|' read -r m c v s tag <<< "$job"
    local name=${m}_${c}_${v}_b45000_${tag}; [ $SMOKE = 1 ] && { tag=smoke32; name=${m}_${c}_${v}_b45000_smoke32; }
    ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && { echo "== $name exists"; continue; }
    echo "== finetune $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$m COND=$c DATASET=sim2sim_${c}_${m} TAG=$tag BUDGETS=45000 VARIANTS=$v FT_STEPS=$FT_STEPS FT_EXTRA="training.seed=$s" \
      NOMINAL_DATASET=$(simds $m) REPLAY_SYSTEM=$(replaysys $m) REPLAY2=sim2sim_nominal_${m}_b45000 $P finetune 2>&1 | grep -aE "^==|optimized|Added a FiLM|Replay|Steps: |Error|Traceback" | cut -c1-260
    echo "-- $name done $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do ft_worker $g & sleep 20; done
wait
echo "== phase F done $(date +%H:%M)"

# ---- phase E: closed loop -------------------------------------------------------------------------------------------
echo '{"xi_fit": [1.0, 0.0, 0.0, 0.0]}' > $BIG/tokA.json; echo '{"xi_fit": [0.0, 0.0, 0.0, 0.0]}' > $BIG/tokS.json
COMMON="sim.terrain=plane sim.total_steps=$TOTAL sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9] batch.seeds=[11,12,13,14,15,16]"
phys() { case $1 in nom) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mot) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  ice) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
base() { [ $1 = P ] && echo go1_paper || echo go1_broadwm; }
cname() { [ $1 = ICE ] && echo friction0.15 || echo very_weak_motor; }
# arm -> checkpoint suffix and token file ("" = model without a token)
sfx() { [ $SMOKE = 1 ] && echo smoke32 || echo $1; }
ck() { local pre=$(base $1)_$(cname $2)
  case $3 in t43A|t43S) echo ${pre}_tok_b45000_$(sfx s43);; t44A|t44S) echo ${pre}_tok_b45000_$(sfx s44);;
    n2) echo ${pre}_dyn_replay2_b45000_$(sfx s42);; t1A) echo ${pre}_tok_noplan_b45000_$(sfx s42);;
    n1) echo ${pre}_dyn_replay_b45000_$(sfx d32s42);; esac; }
tokfile() { case $1 in t43A|t44A|t1A) echo $BIG/tokA.json;; t43S|t44S) echo $BIG/tokS.json;; *) echo "";; esac; }
own() { [ $1 = ICE ] && echo ice || echo mot; }; oth() { [ $1 = ICE ] && echo mot || echo ice; }
FILT="Traceback|rror:|falls written|diagnostics written|FiLM context|Killed|OOM|RESOURCE_EXHAUSTED"
Q=$BIG/queue.txt; : > $Q
AD="ICE MOT"; MS="P B"; [ $SMOKE = 1 ] && { AD=ICE; MS=P; }
add() { local arm=$1 which=$2; for a in $AD; do for m in $MS; do local c; case $which in own) c=$(own $a);; oth) c=$(oth $a);; nom) c=nom;; esac
  echo "${m}_${a}${arm}_${c}|$m|$a|$arm|$c" >> $Q; done; done; }
if [ $SMOKE = 1 ]; then add n2 own; add t43A own
else
  add n2 own; add n2 oth                                    # ingredient: no token, both replay sets
  add t43A own; add t43S oth; add t44A own; add t44S oth    # replication of G1 / G2
  add n1 own; add n1 oth                                    # ingredient: design-20 recipe under this protocol
  add t1A own; add t1A oth                                  # ingredient: token without simulation planner windows
  add t43A oth; add t44A oth                                # retention without a switch (R3)
  add n2 nom
fi
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name m a arm c <<< "$job"
    [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; continue; }
    local tf; tf=$(tokfile $arm)
    echo "== $name gpu$g $(ck $m $a $arm) $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$(ck $m $a $arm) control.prior_only=false ${tf:+sim.xi_json=$tf} \
      $(phys $c) $COMMON output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz \
      output.model_name=$name output.condition=$c --headless 2>&1 | grep -aE "$FILT" | head -6
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage32 done $(date)"
