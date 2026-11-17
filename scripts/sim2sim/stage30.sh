#!/bin/bash
# Design doc 30 (docs/design30_support_signal_ko.md): offline checks before a support-aware planner.
#   F  three more fine-tuning seeds (44-46) of the SimDist recipe per model x condition -> 5 adapted members each
#   C  in-coverage planner data for the broad-DR model (nominal, 45k steps), the paper model's set exists
#   S  support signals (stage30_signals.py), A action adapter (stage30_adapter.py), 2 models x 2 adaptation conditions
# usage (gpu1): tmux new -d -s s30 'bash scripts/sim2sim/stage30.sh > results/stage30_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage30; mkdir -p $BIG
FILT="^==|RESULT|written|windows|Traceback|rror:|Killed|RESOURCE_EXHAUSTED|step 2999"
simds() { [ "$1" = go1_broadwm ] && echo 2026-11-05_11-39-08 || echo 2026-10-02_01-18-59; }
tag() { [ "$1" = go1_paper ] && echo P || echo B; }
ctag() { [ "$1" = friction0.15 ] && echo ice || echo mot; }
other() { [ "$1" = friction0.15 ] && echo very_weak_motor || echo friction0.15; }
echo "### stage30 start $(date)"

finetune() { local g=$1 m=$2 c=$3 s=$4; local name=${m}_${c}_dyn_b45000_s$s
  ls -d checkpoints/models/$name/[0-9]* >/dev/null 2>&1 && { echo "== $name exists"; return; }
  echo "== finetune $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" BASE=$m RESUME_CKPT=$m COND=$c DATASET=sim2sim_${c}_${m} TAG=s$s BUDGETS=45000 VARIANTS=dyn FT_STEPS=3000 FT_EXTRA="training.seed=$s" \
    NOMINAL_DATASET=$(simds $m) $P finetune 2>&1 | grep -E "^==|Steps: |Error|Traceback" | cut -c1-200; }
gpu_chain() { local g=$1; shift; local j; for j in "$@"; do IFS='|' read -r m c s <<< "$j"; finetune $g $m $c $s; done; }
# phase F: 12 fine-tunes, 3 per GPU; phase C runs on GPU 0 after its chain
gpu_chain 0 "go1_paper|friction0.15|44" "go1_paper|friction0.15|45" "go1_paper|friction0.15|46" &
gpu_chain 1 "go1_paper|very_weak_motor|44" "go1_paper|very_weak_motor|45" "go1_paper|very_weak_motor|46" &
gpu_chain 2 "go1_broadwm|friction0.15|44" "go1_broadwm|friction0.15|45" "go1_broadwm|friction0.15|46" &
gpu_chain 3 "go1_broadwm|very_weak_motor|44" "go1_broadwm|very_weak_motor|45" "go1_broadwm|very_weak_motor|46" &
wait
echo "== phase F done $(date +%H:%M)"
NEG=sim2sim_nominal_go1_broadwm
if [ ! -d datasets/real/${NEG}_b45000/processed_data_go1_H-25_T-25 ]; then
  echo "== collect $NEG $(date +%H:%M)"
  [ -d datasets/real/$NEG/raw ] || GPUS='"device=0"' BASE=go1_broadwm COND=nominal DATASET=$NEG COLLECT_CKPT=go1_broadwm ROUNDS=3 COLLECT_STEPS=5000 $P collect 2>&1 | grep -E "^==|Error|Traceback" | cut -c1-160
  GPUS='"device=0"' DATASET=$NEG BUDGETS=45000 $P process 2>&1 | grep -E "^==|Error|Traceback|Total" | cut -c1-160
fi
echo "== phase C done $(date +%H:%M)"

signals() { local g=$1 m=$2 c=$3; local n=sig_$(tag $m)_$(ctag $c); local mem=""; local s
  for s in 42 43 44 45 46; do mem="$mem ${m}_${c}_dyn_b45000_s$s"; done
  echo "== $n gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage30_signals.py --base $m --members $mem --train sim2sim_${c}_${m}_b45000 --neg sim2sim_nominal_${m}_b45000 \
    --own sim2sim_${c}_${m}_ho_b15000 --others nominal=sim2sim_nominal_${m}_ho_b15000 other=sim2sim_$(other $c)_${m}_ho_b15000 payload=sim2sim_payload_${m}_ho_b15000 \
    --out $BIG/$n 2>&1 | grep -aE "$FILT" | cut -c1-200
  echo "-- $n done $(date +%H:%M)"; }
adapter() { local g=$1 m=$2 c=$3; local n=adp_$(tag $m)_$(ctag $c)
  echo "== $n gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage30_adapter.py --base $m --ft ${m}_${c}_dyn_b45000_s42 --train sim2sim_${c}_${m}_b45000 \
    --evals own=sim2sim_${c}_${m}_ho_b15000 nominal=sim2sim_nominal_${m}_ho_b15000 other=sim2sim_$(other $c)_${m}_ho_b15000 payload=sim2sim_payload_${m}_ho_b15000 \
    --out $BIG/$n 2>&1 | grep -aE "$FILT" | cut -c1-200
  echo "-- $n done $(date +%H:%M)"; }
both() { signals $1 $2 $3; adapter $1 $2 $3; }
both 0 go1_paper friction0.15 & sleep 20
both 1 go1_paper very_weak_motor & sleep 20
both 2 go1_broadwm friction0.15 & sleep 20
both 3 go1_broadwm very_weak_motor &
wait
echo "### stage30 done $(date)"
