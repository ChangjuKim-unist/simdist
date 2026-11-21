#!/bin/bash
# Design doc 36 (docs/design36_sampling_checks_ko.md): C1 warm-start fix (init_mode shift vs shift_fwd, five planner cells) and
# C3 candidate rank check (branch_eval group M). C2 runs locally on the design-34 logs (stage36_logs.py, no GPU).
# usage (gpu1, after design 35 has finished): tmux new -d -s s36 'bash scripts/sim2sim/stage36.sh > /mnt/newssd/cjkim/simdist/results/stage36_server.log 2>&1'
# SMOKE=1: local check (one GPU, 150-step evaluations)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
export CACHE_DIR=${CACHE_DIR:-/mnt/newssd/cjkim/isaac-cache}
MIN_ROOT_KB=${MIN_ROOT_KB:-400000}
BIG=${BIG:-/mnt/newssd/cjkim/simdist/results/stage36}; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; SMOKE=${SMOKE:-0}; M=go1_paper
TOTAL=5000; SEEDS="[11,12,13,14,15,16]"; WAIT=${WAIT:-s35}
if [ $SMOKE = 1 ]; then TOTAL=150; SEEDS="[11]"; WAIT=""; fi
phys() { [ $1 = nominal ] && return; BASE=$M COND=$1 bash -c "source <(sed -n '/^condition_overrides()/,/^}/p' $P); condition_overrides $1"; }
FILT="Traceback|rror:|Prior samples|falls written|diagnostics written|Killed|OOM|RESOURCE_EXHAUSTED"
LOCK=$BIG/queue.lock
evalrun() { local g=$1 name=$2 ck=$3 c=$4 extra=$5
  [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; return; }
  echo "== $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ck $extra $(phys $c) sim.terrain=plane \
    sim.episode_length_s=100 "batch.speeds=[0.3,0.6,0.9]" "batch.seeds=$SEEDS" sim.total_steps=$TOTAL output.csv=$BIG/eval_$name.csv \
    output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz output.model_name=$name output.condition=$c --headless 2>&1 \
    | grep -aE "$FILT" | grep -v cuStreamGetGreenCtx | head -8
  echo "-- $name done gpu$g $(date +%H:%M)"; }
# C3: anchor run (the planner walks, candidates recorded at every snapshot), then the branch run of group M on a seeded
# subset of C3_POINTS branch points
C3_POINTS=${C3_POINTS:-96}
c3run() { local g=$1 name=$2 ck=$3 c=$4
  [ -f $BIG/${name}_branch.done ] && { echo "== $name exists"; return; }
  echo "== $name gpu$g $(date +%H:%M)"; local an=$BIG/${name}_anchor.npz sm=""
  [ $SMOKE = 1 ] && sm="branch.m_counts=[2,1,2,1,0]"
  [ -f $an ] || GPUS="\"device=$g\"" $RUN python scripts/sim2sim/branch_eval.py mode=anchor model.checkpoint=$ck control.prior_only=false \
    branch.drive=planner branch.m_record=true branch.every=50 branch.warmup=300 $sm $(phys $c) sim.terrain=plane sim.episode_length_s=100 \
    sim.total_steps=$TOTAL "batch.speeds=[0.3,0.6,0.9]" "batch.seeds=$SEEDS" output.condition=$c branch.out=$an --headless 2>&1 \
    | grep -aE "Traceback|rror:|RESULT|snapshot|Killed|OOM" | grep -v cuStreamGetGreenCtx | tail -4
  [ -f $an ] || { echo "!! $name anchor failed"; return; }
  local np=$C3_POINTS; [ $SMOKE = 1 ] && np=2
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$ck control.prior_only=true $(phys $c) \
    sim.terrain=plane sim.episode_length_s=100 "branch.groups=[M]" branch.W=8 branch.m_clones_mean_head=3 branch.sample_points=$np \
    output.condition=$c branch.anchor=$an branch.csv=$BIG/${name}_branch.csv --headless 2>&1 \
    | grep -aE "Traceback|rror:|RESULT|round|Killed|OOM" | grep -v cuStreamGetGreenCtx | tail -3
  [ -s $BIG/${name}_branch.csv ] && touch $BIG/${name}_branch.done
  echo "-- $name done gpu$g $(date +%H:%M)"; }
dojob() { local g=$1 r=$2 kind name ck c extra
  kind=${r%%|*}; r=${r#*|}; name=${r%%|*}; r=${r#*|}; ck=${r%%|*}; r=${r#*|}; c=${r%%|*}; extra=${r#*|}
  case $kind in ev) evalrun $g $name $ck $c "$extra";; c3) c3run $g $name $ck $c;; esac; }
worker() { local g=$1 Q=$2
  while true; do
    [ "$(df --output=avail / | tail -1)" -lt $MIN_ROOT_KB ] && { echo "!! root disk nearly full, gpu$g stops taking jobs $(date +%H:%M)"; break; }
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$job" ] && break
    dojob $g "$job"
  done; }
runqueue() { for g in $GPU_LIST; do worker $g $1 & sleep 20; done; wait; }

if [ -n "$WAIT" ]; then while tmux has-session -t $WAIT 2>/dev/null; do sleep 60; done; fi
echo "### stage36 start $(date)"; git log --oneline -1 | cut -c1-60

Q=$BIG/queue.txt; : > $Q
# C1: five planner cells x two warm-start modes (both rerun so that ESS / policy share are recorded for both)
for mode in shift fwd; do
  im=$mode; [ $mode = fwd ] && im=shift_fwd
  echo "ev|c1_nominal_PRE_$mode|$M|nominal|control.prior_only=false control.init_mode=$im" >> $Q
  for c in tow20 mot50; do
    echo "ev|c1_${c}_PRE_$mode|$M|$c|control.prior_only=false control.init_mode=$im" >> $Q
    echo "ev|c1_${c}_ADP_$mode|${M}_${c}_dyn_b45000_s42|$c|control.prior_only=false control.init_mode=$im" >> $Q
  done
done
[ $SMOKE = 1 ] && sed -i "s#${M}_\(tow20\|mot50\)_dyn_b45000_s42#${M}_tow20_dyn_b1200_smk#" $Q
# C3 first in the queue (longest): adapted models on their own condition, paper model on nominal
Q3=$BIG/queue3.txt; : > $Q3
echo "c3|c3_nominal|$M|nominal|" >> $Q3
for c in tow20 mot50 legR50; do echo "c3|c3_$c|${M}_${c}_dyn_b45000_s42|$c|" >> $Q3; done
[ $SMOKE = 1 ] && { echo "c3|c3_tow20|${M}_tow20_dyn_b1200_smk|tow20|" > $Q3; }
cat $Q >> $Q3; mv $Q3 $Q
runqueue $Q
echo "### stage36 done $(date)"
