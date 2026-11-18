#!/bin/bash
# Design doc 31 (docs/design31_token_gate_ko.md): gate for the condition-token world model.
#   F  token fine-tunes (variant tok: FiLM token [1,0,0,0] on the adaptation batch, xi = 0 on replay; replay = pretraining
#      windows + simulation planner windows), 2 models x 2 adaptation conditions, seed 42, SimDist budget (3000 steps)
#   S  offline token-error-difference signal (stage31_signals.py), G3
#   E  closed loop (design-29 protocol: 18 robots x 5000 steps) with the adapted token (tA) or the simulation token (tS):
#      priority order  tA on own (G1), tS on other (G2), tS on own, tA on other, tS on nominal
# usage (gpu1): tmux new -d -s s31 'bash scripts/sim2sim/stage31.sh > results/stage31_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage31; mkdir -p $BIG
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
simds() { [ "$1" = go1_broadwm ] && echo 2026-11-05_11-39-08 || echo paper_sub8192; }
replaysys() { [ "$1" = go1_broadwm ] && echo go1dr || echo ""; }
tag() { [ "$1" = go1_paper ] && echo P || echo B; }
ctag() { [ "$1" = friction0.15 ] && echo ice || echo mot; }
other() { [ "$1" = friction0.15 ] && echo very_weak_motor || echo friction0.15; }
echo "### stage31 start $(date)"

finetune() { local g=$1 m=$2 c=$3; local name=${m}_${c}_tok_b45000_s42
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && { echo "== $name exists"; return; }
  echo "== finetune $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$m COND=$c DATASET=sim2sim_${c}_${m} TAG=s42 BUDGETS=45000 VARIANTS=tok FT_STEPS=3000 FT_EXTRA="training.seed=42" \
    NOMINAL_DATASET=$(simds $m) REPLAY_SYSTEM=$(replaysys $m) REPLAY2=sim2sim_nominal_${m}_b45000 $P finetune 2>&1 | grep -aE "^==|optimized|Added a FiLM|Replay|Steps: |Error|Traceback" | cut -c1-260
  echo "-- $name done $(date +%H:%M)"; }
signals() { local g=$1 m=$2 c=$3; local n=sig_$(tag $m)_$(ctag $c)
  echo "== $n gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage31_signals.py --base $m --dyn ${m}_${c}_dyn_b45000_s42 --tok ${m}_${c}_tok_b45000_s42 \
    --own sim2sim_${c}_${m}_ho_b15000 --others nominal=sim2sim_nominal_${m}_ho_b15000 other=sim2sim_$(other $c)_${m}_ho_b15000 payload=sim2sim_payload_${m}_ho_b15000 \
    --out $BIG/$n 2>&1 | grep -aE "^==|written|windows|Traceback|rror:|Killed|RESOURCE_EXHAUSTED" | cut -c1-220
  echo "-- $n done $(date +%H:%M)"; }
fs() { finetune $1 $2 $3; signals $1 $2 $3; }
fs 0 go1_paper friction0.15 & sleep 20
fs 1 go1_paper very_weak_motor & sleep 20
fs 2 go1_broadwm friction0.15 & sleep 20
fs 3 go1_broadwm very_weak_motor &
wait
echo "== phases F, S done $(date +%H:%M)"

# ---- phase E: closed loop -------------------------------------------------------------------------------------------
echo '{"xi_fit": [1.0, 0.0, 0.0, 0.0]}' > $BIG/tokA.json; echo '{"xi_fit": [0.0, 0.0, 0.0, 0.0]}' > $BIG/tokS.json
COMMON="sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 batch.speeds=[0.3,0.6,0.9] batch.seeds=[11,12,13,14,15,16]"
phys() { case $1 in nom) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  mot) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  ice) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
base() { [ $1 = P ] && echo go1_paper || echo go1_broadwm; }
ck() { case $2 in ICE) echo $(base $1)_friction0.15_tok_b45000_s42;; MOT) echo $(base $1)_very_weak_motor_tok_b45000_s42;; esac; }
own() { [ $1 = ICE ] && echo ice || echo mot; }; oth() { [ $1 = ICE ] && echo mot || echo ice; }
FILT="Traceback|rror:|falls written|diagnostics written|FiLM context|Killed|OOM|RESOURCE_EXHAUSTED"
Q=$BIG/queue.txt; LOCK=$BIG/queue.lock; : > $Q
# name|model|adaptation|token|condition
for a in ICE MOT; do for m in P B; do echo "${m}_${a}tA_$(own $a)|$m|$a|A|$(own $a)" >> $Q; done; done   # G1
for a in ICE MOT; do for m in P B; do echo "${m}_${a}tS_$(oth $a)|$m|$a|S|$(oth $a)" >> $Q; done; done   # G2
for a in ICE MOT; do for m in P B; do echo "${m}_${a}tS_$(own $a)|$m|$a|S|$(own $a)" >> $Q; done; done
for a in ICE MOT; do for m in P B; do echo "${m}_${a}tA_$(oth $a)|$m|$a|A|$(oth $a)" >> $Q; done; done
for a in ICE MOT; do for m in P B; do echo "${m}_${a}tS_nom|$m|$a|S|nom" >> $Q; done; done

worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name m a t c <<< "$job"
    [ -f $BIG/eval_$name.csv ] && { echo "== $name exists"; continue; }
    echo "== $name gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$(ck $m $a) control.prior_only=false sim.xi_json=$BIG/tok$t.json \
      $(phys $c) $COMMON output.csv=$BIG/eval_$name.csv output.falls_csv=$BIG/falls_$name.csv output.diag_npz=$BIG/diag_$name.npz \
      output.model_name=$name output.condition=$c --headless 2>&1 | grep -aE "$FILT" | head -6
    echo "-- $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage31 done $(date)"
