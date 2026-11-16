#!/bin/bash
# Design doc 27 stage 1 (docs/design27_recovery_quality_ko.md): recovery after slips (branching) + motion-quality runs.
#   wave A: 4 slip-triggered anchor runs (broad / paper x planner-driven / policy-driven with shadow planner), ~55 min
#   wave B: 4 branch runs from the anchors (A0 policy head, Fcl planner, Nrep replayed planner deviation) + 4 normal-floor runs
#   usage (gpu1): tmux new -d -s s27 'bash scripts/sim2sim/stage27.sh > results/stage27_server.log 2>&1'
# Outputs on the large disk (root disk nearly full): /mnt/newssd/cjkim/simdist/results/stage27 (mounted in the container).
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage27; mkdir -p $BIG
IN=/workspace/simdist/results/stage27   # committed inputs: dev_bank.npz, noise_params.json
GPU_LIST=${GPU_LIST:-"0 1 2 3"}
THETA=0.9553579986095428
SEEDS='batch.seeds=[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]'
COMMON="sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 sim.cmd_ramp_steps=50 batch.speeds=[0.9]"
FILT="Traceback|rror:|falls written|diagnostics written|branch points written|slip trigger|slip events|RESULT branch|Nrep deviation"

anchor() { local g=$1 name=$2 ckpt=$3 drive=$4
  local prior=true; [ "$drive" = planner ] && prior=false
  echo "== anchor $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/branch_eval.py mode=anchor model.checkpoint=$ckpt control.prior_only=$prior \
    branch.drive=$drive branch.trigger=slip branch.slip_theta=$THETA branch.min_since=300 branch.ctrl_every=100 branch.ctrl_quiet=50 \
    branch.horizon=150 branch.out=$BIG/anchor_$name.npz sim.friction=0.15 $COMMON "$SEEDS" \
    output.csv=$BIG/eval_anchor_gpu$g.csv output.falls_csv=$BIG/falls_anchor_gpu$g.csv output.diag_npz=$BIG/diag_$name.npz \
    output.model_name=$name output.condition=friction0.15 --headless 2>&1 | grep -aE "$FILT" | head -6
  echo "-- anchor $name done gpu$g $(date +%H:%M)"; }

echo "### stage27 start $(date)"
anchor 0 AB_P go1_broadwm planner &
sleep 20; anchor 1 AB_pi go1_broadwm policy &
sleep 20; anchor 2 AP_P go1_paper planner &
sleep 20; anchor 3 AP_pi go1_paper policy &
wait
echo "### wave A done $(date)"

Q=$BIG/queue.txt; LOCK=$BIG/queue.lock; : > $Q
# kind|name|ckpt|extra ; branch runs first (main experiment), then the normal-floor runs (planner before policy)
echo "branch|AB_P|go1_broadwm|broad" >> $Q
echo "branch|AB_pi|go1_broadwm|broad" >> $Q
echo "branch|AP_P|go1_paper|paper" >> $Q
echo "branch|AP_pi|go1_paper|paper" >> $Q
echo "floor|NB_P|go1_broadwm|control.prior_only=false" >> $Q
echo "floor|NP_P|go1_paper|control.prior_only=false" >> $Q
echo "floor|NB_pi|go1_broadwm|control.prior_only=true" >> $Q
echo "floor|NP_pi|go1_paper|control.prior_only=true" >> $Q
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r kind name ckpt extra <<< "$job"
    echo "== $kind $name gpu$g $(date +%H:%M)"
    if [ "$kind" = branch ]; then
      GPUS="\"device=$g\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$ckpt control.prior_only=true \
        sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0 sim.terrain=plane sim.episode_length_s=100 \
        branch.anchor=$BIG/anchor_$name.npz branch.csv=$BIG/branch_$name.csv branch.steps_dir=$BIG/steps_$name \
        'branch.groups=[A0,Fcl,Nrep]' branch.chunk=150 branch.horizon=150 branch.W=8 branch.J=3 branch.max_events=200 branch.max_ctrl=80 \
        branch.dev_bank=$IN/dev_bank.npz branch.noise_key=$extra \
        output.model_name=$name output.condition=friction0.15 --headless 2>&1 | grep -aE "$FILT|round (1|[0-9]*0)/" | head -20
    else
      GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $extra sim.friction=0.8 $COMMON "$SEEDS" \
        output.csv=$BIG/eval_floor_gpu$g.csv output.falls_csv=$BIG/falls_floor_gpu$g.csv output.diag_npz=$BIG/diag_$name.npz \
        output.model_name=$name output.condition=friction0.8 --headless 2>&1 | grep -aE "$FILT" | head -4
    fi
    echo "-- $kind $name done gpu$g $(date +%H:%M)"
  done; }
for g in $GPU_LIST; do worker $g & sleep 20; done
wait
echo "### stage27 done $(date)"
