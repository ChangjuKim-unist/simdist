#!/bin/bash
# Design doc 35 (docs/design35_headroom_gates_ko.md): cheap gates before building a behaviour-side method on the design-34 conditions.
#   P1  integral term on the model-side command (sim.cmd_ki): policy head alone and planner, pretrained and adapted models
#   P2  existing policy heads (prior.head_from): each head alone on nominal + 4 levels; then the best head per level on the
#       adapted dynamics inside the planner
#   P3  specialist experts: the paper expert fine-tuned inside each level (SIMDIST_COND), against the paper expert and a
#       nominal continuation (b0c) of the same length
#   P4  single-style experts: the paper expert fine-tuned at nominal with one posture target (GO1_STYLE_GRAV), zero-shot
# Levels = design-34 choices without delay: legR50, slope5, tow20, mot50. 18 robots x 5000 steps, flat ground.
# usage (gpu1): tmux new -d -s s35 'bash scripts/sim2sim/stage35.sh > /mnt/newssd/cjkim/simdist/results/stage35_server.log 2>&1'
# SMOKE=1: local check (one GPU, tiny RL runs, 150-step evaluations, two heads, one level)
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
# Isaac Sim caches grow by about 1.2 GB per GPU and filled gpu1's root disk on 2026-11-21 (RL scenes); keep them on the big disk
export CACHE_DIR=${CACHE_DIR:-/mnt/newssd/cjkim/isaac-cache}
MIN_ROOT_KB=${MIN_ROOT_KB:-400000}  # stop taking jobs when the root disk (docker layers, /tmp) has less than this free
BIG=${BIG:-/mnt/newssd/cjkim/simdist/results/stage35}; S34=${S34:-/mnt/newssd/cjkim/simdist/results/stage34}
RLD=${RLD:-/mnt/newssd/cjkim/simdist/rl}; mkdir -p $BIG $RLD
GPU_LIST=${GPU_LIST:-"0 1 2 3"}; SMOKE=${SMOKE:-0}; M=go1_paper
PAPER_RL=2026-10-01_10-38-36; PAPER_IT=4999
CONDS="legR50 slope5 tow20 mot50"
HEADS="go1_paper_broadhead go1_paper_targethead go1_paper_targethead_s2 go1_paper_payloadhead go1_paper_fr_blanket go1_paper_rw_t05 go1_paper_rw_t06 go1_paper_rw_w02 go1_paper_rw_w06 go1_broadwm"
TOTAL=5000; SEEDS="[11,12,13,14,15,16]"; KI=0.5; RL_IT=1500; RL_ENVS=${RL_ENVS:-1024}; RL_SAVE=500
RLJOBS="b0c $(for c in $CONDS; do echo spec_$c; done) sty_pp sty_pm sty_rp sty_rm"
# gpu0 joins only after design 34 has finished (its last adaptation runs there)
WAIT0=${WAIT0:-s34}  # tmux session of design 34
if [ $SMOKE = 1 ]; then
  CONDS="tow20"; HEADS="go1_paper_broadhead go1_broadwm"; TOTAL=150; SEEDS="[11]"; RL_IT=2; RL_ENVS=64; RL_SAVE=1; WAIT0=""
  RLJOBS="b0c spec_tow20 sty_pp"
fi
phys() { [ $1 = nominal ] && return; BASE=$M COND=$1 bash -c "source <(sed -n '/^condition_overrides()/,/^}/p' $P); condition_overrides $1"; }
FILT="Traceback|rror:|^condition:|^integral|Prior samples|Acting with|falls written|diagnostics written|Killed|OOM|RESOURCE_EXHAUSTED"
LOCK=$BIG/queue.lock
evalrun() { local g=$1 name=$2 ck=$3 c=$4 extra=$5
  [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; return; }
  echo "== $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN env SIMDIST_RL_DIR=$RLD python scripts/simulate_batch.py model.checkpoint=$ck $extra $(phys $c) sim.terrain=plane \
    sim.episode_length_s=100 "batch.speeds=[0.3,0.6,0.9]" "batch.seeds=$SEEDS" sim.total_steps=$TOTAL output.csv=$BIG/eval_$name.csv \
    output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz output.model_name=$name output.condition=$c --headless 2>&1 \
    | grep -aE "$FILT" | grep -v cuStreamGetGreenCtx | head -8
  echo "-- $name done gpu$g $(date +%H:%M)"; }
cond_json() { case $1 in
  legR50) echo '{"leg_strength":[1,1,0.5,1]}';; slope5) echo '{"friction":0.3,"slope_deg":5}';;
  tow20) echo '{"tow_force":20}';; mot50) echo '{"motor_strength":0.5}';; *) echo "";; esac; }
style_grav() { case $1 in sty_pp) echo "0.15,0";; sty_pm) echo "-0.15,0";; sty_rp) echo "0,0.15";; sty_rm) echo "0,-0.15";; *) echo "";; esac; }
rltrain() { local g=$1 job=$2
  grep -q "^$job " $BIG/rl_runs.txt 2>/dev/null && { echo "== rl $job exists"; return; }
  local cj="" sg=""; case $job in spec_*) cj=$(cond_json ${job#spec_});; sty_*) sg=$(style_grav $job);; esac
  echo "== rl $job gpu$g cond '$cj' style '$sg' $(date +%H:%M)"
  local log=$BIG/rl_$job.log
  GPUS="\"device=$g\"" $RUN env SIMDIST_RL_DIR=$RLD SIMDIST_COND="$cj" GO1_STYLE_GRAV="$sg" python scripts/train_rl.py --task Go1 --headless \
    --num_envs $RL_ENVS --seed 1 --max_iterations $RL_IT --resume True --load_run $PAPER_RL --checkpoint model_$PAPER_IT.pt \
    agent.save_interval=$RL_SAVE > $log 2>&1
  local run; run=$(grep -aoE "Logging experiment in directory: .*" $log | head -1 | sed 's#.*/##')
  grep -aE "Traceback|^condition:|posture style|Learning iteration [0-9]+/" $log | grep -v cuStreamGetGreenCtx | tail -4 | cut -c1-200
  [ -n "$run" ] && ls $RLD/$run/model_*.pt >/dev/null 2>&1 || { echo "!! rl $job produced no checkpoint ($log)"; return; }
  GPUS="\"device=$g\"" $RUN env SIMDIST_RL_DIR=$RLD python scripts/export_policies.py --task Go1Play -r $run --headless 2>&1 | grep -aE "Traceback|rror:" | grep -v cuStreamGetGreenCtx | head -3
  local it; it=$(ls $RLD/$run/policies/ | sed -n 's/policy_\([0-9]*\)\.pt/\1/p' | sort -n | tail -1)
  flock $LOCK bash -c "echo '$job $run $it' >> $BIG/rl_runs.txt"
  echo "-- rl $job done: run $run iteration $it $(date +%H:%M)"; }
dojob() { local g=$1 r=$2 kind name ck c extra  # split without a here-string (needs no temp file)
  kind=${r%%|*}; r=${r#*|}; name=${r%%|*}; r=${r#*|}; ck=${r%%|*}; r=${r#*|}; c=${r%%|*}; extra=${r#*|}
  if [ "$kind" = rl ]; then rltrain $g $name; else evalrun $g $name $ck $c "$extra"; fi; }
worker() { local g=$1 Q=$2
  if [ $g = 0 ] && [ -n "$WAIT0" ]; then while tmux has-session -t $WAIT0 2>/dev/null; do sleep 60; done; echo "== gpu0 joins $(date +%H:%M)"; fi
  while true; do
    [ "$(df --output=avail / | tail -1)" -lt $MIN_ROOT_KB ] && { echo "!! root disk nearly full, gpu$g stops taking jobs $(date +%H:%M)"; break; }
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q"); [ -z "$job" ] && break
    dojob $g "$job"
  done; }
runqueue() { for g in $GPU_LIST; do worker $g $1 & sleep 20; done; wait; }
echo "### stage35 start $(date)"

# the paper expert where training and evaluation look for RL runs
if [ ! -f $RLD/$PAPER_RL/model_$PAPER_IT.pt ]; then
  mkdir -p $RLD/$PAPER_RL/policies; cp checkpoints/rl/$PAPER_RL/model_$PAPER_IT.pt $RLD/$PAPER_RL/ 2>/dev/null
  cp -r checkpoints/rl/$PAPER_RL/params $RLD/$PAPER_RL/ 2>/dev/null; cp checkpoints/rl/$PAPER_RL/policies/policy_$PAPER_IT.pt $RLD/$PAPER_RL/policies/
fi
[ -f $RLD/$PAPER_RL/model_$PAPER_IT.pt ] && [ -f $RLD/$PAPER_RL/policies/policy_$PAPER_IT.pt ] || { echo "!! paper expert missing in $RLD"; exit 1; }
for h in $HEADS; do [ -d checkpoints/models/$h ] || { echo "!! head $h missing"; exit 1; }; done

# ---- stage 1: RL fine-tunes (longest first), P1, P2 heads alone (queue line: kind|name|checkpoint|condition|extra) -----
Q=$BIG/queue1.txt; : > $Q
for j in $RLJOBS; do echo "rl|$j|||" >> $Q; done
echo "ev|nominal_PREI|$M|nominal|control.prior_only=false sim.cmd_ki=$KI" >> $Q
for c in $CONDS; do echo "ev|${c}_ADPI|${M}_${c}_dyn_b45000_s42|$c|control.prior_only=false sim.cmd_ki=$KI" >> $Q; done
for c in nominal $CONDS; do echo "ev|${c}_POLI|$M|$c|control.prior_only=true sim.cmd_ki=$KI" >> $Q; done
for c in nominal $CONDS; do for h in $HEADS; do echo "ev|${c}_H_${h}|$M|$c|control.prior_only=true prior.head_from=$h" >> $Q; done; done
[ $SMOKE = 1 ] && sed -i "s#${M}_tow20_dyn_b45000_s42#${M}_tow20_dyn_b1200_smk#" $Q
runqueue $Q
echo "== stage 1 done $(date +%H:%M)"

# ---- stage 2: expert evaluations (P3, P4) and the best head per level inside the planner (P2) -------------------------
Q=$BIG/queue2.txt; : > $Q
echo "paper $PAPER_RL $PAPER_IT" > $BIG/experts.txt; cat $BIG/rl_runs.txt >> $BIG/experts.txt 2>/dev/null
while read -r job run it; do
  case $job in spec_*) ev="nominal ${job#spec_}";; *) ev="nominal $CONDS";; esac
  for c in $ev; do echo "ev|${c}_X_${job}|$M|$c|policy.rl_run=$run policy.iteration=$it" >> $Q; done
done < $BIG/experts.txt
GPUS="\"device=$(echo $GPU_LIST | cut -d' ' -f1)\"" $RUN python scripts/sim2sim/stage35_score.py --d $BIG --d34 $S34 --conds "$CONDS" \
  --heads "$HEADS" --pick_heads 2>&1 | grep -aE "^best|rror|Traceback"
if [ -s $BIG/best_heads.txt ]; then
  while read -r c h; do
    ck=${M}_${c}_dyn_b45000_s42; [ $SMOKE = 1 ] && ck=${M}_${c}_dyn_b1200_smk
    echo "ev|${c}_ADPH|$ck|$c|control.prior_only=false prior.head_from=$h" >> $Q
  done < $BIG/best_heads.txt
fi
WAIT0=""; runqueue $Q
echo "### stage35 done $(date)"
