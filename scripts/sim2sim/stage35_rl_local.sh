#!/bin/bash
# Design doc 35, P3/P4 RL fine-tunes on the local RTX 3060. gpu1's RTX 2080 Ti fails at the ray-caster init with
# "Warp CUDA error 2: out of memory" at 1024, 768 and 512 envs (also with the Warp mempool off), and every earlier RL run
# was trained locally. Same settings as stage35.sh rltrain (1024 envs, seed 1, 1500 iterations from the paper expert).
# Each finished run is copied to gpu1 and recorded in gpu1's rl_runs.txt; stage35.sh with RL_REMOTE=1 evaluates it there.
# usage (local): nohup bash scripts/sim2sim/stage35_rl_local.sh > results/stage35_rl_local.log 2>&1 &
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
RLD=$PWD/checkpoints/rl; LOGD=$PWD/results/stage35_rl_local; mkdir -p $LOGD
REMOTE_RLD=/mnt/newssd/cjkim/simdist/rl; REMOTE_BIG=/mnt/newssd/cjkim/simdist/results/stage35
PAPER_RL=2026-10-01_10-38-36; PAPER_IT=4999; RL_IT=${RL_IT:-1500}; RL_ENVS=${RL_ENVS:-1024}; RL_SAVE=500
RLJOBS=${RLJOBS:-"b0c spec_legR50 spec_slope5 spec_tow20 spec_mot50 sty_pp sty_pm sty_rp sty_rm"}
# same maps as stage35.sh
cond_json() { case $1 in
  legR50) echo '{"leg_strength":[1,1,0.5,1]}';; slope5) echo '{"friction":0.3,"slope_deg":5}';;
  tow20) echo '{"tow_force":20}';; mot50) echo '{"motor_strength":0.5}';; *) echo "";; esac; }
style_grav() { case $1 in sty_pp) echo "0.15,0";; sty_pm) echo "-0.15,0";; sty_rp) echo "0,0.15";; sty_rm) echo "0,-0.15";; *) echo "";; esac; }
remote() { ssh gpu1 "$1" 2>&1 | grep -v Ubuntu; }
echo "### stage35 local RL start $(date)"
remote "mkdir -p $REMOTE_BIG"
for job in $RLJOBS; do
  remote "grep -q '^$job ' $REMOTE_BIG/rl_runs.txt 2>/dev/null && echo yes" | grep -q yes && { echo "== rl $job exists on gpu1"; continue; }
  cj=""; sg=""; case $job in spec_*) cj=$(cond_json ${job#spec_});; sty_*) sg=$(style_grav $job);; esac
  echo "== rl $job cond '$cj' style '$sg' $(date +%H:%M)"; log=$LOGD/rl_$job.log
  GPUS='"device=0"' $RUN env SIMDIST_COND="$cj" GO1_STYLE_GRAV="$sg" python scripts/train_rl.py --task Go1 --headless \
    --num_envs $RL_ENVS --seed 1 --max_iterations $RL_IT --resume True --load_run $PAPER_RL --checkpoint model_$PAPER_IT.pt \
    agent.save_interval=$RL_SAVE > $log 2>&1
  run=$(grep -aoE "Logging experiment in directory: .*" $log | head -1 | sed 's#.*/##')
  grep -aE "Traceback|^condition:|posture style|Learning iteration [0-9]+/" $log | grep -v cuStreamGetGreenCtx | tail -3 | cut -c1-200
  if [ -z "$run" ] || ! ls $RLD/$run/model_*.pt >/dev/null 2>&1; then
    echo "!! rl $job produced no checkpoint ($log)"; remote "echo '$job' >> $REMOTE_BIG/rl_failed.txt"; continue; fi
  GPUS='"device=0"' $RUN python scripts/export_policies.py --task Go1Play -r $run --headless 2>&1 | grep -aE "Traceback|rror:" | grep -v cuStreamGetGreenCtx | head -3
  it=$(ls $RLD/$run/policies/ | sed -n 's/policy_\([0-9]*\)\.pt/\1/p' | sort -n | tail -1)
  [ -z "$it" ] && { echo "!! rl $job export failed"; remote "echo '$job' >> $REMOTE_BIG/rl_failed.txt"; continue; }
  rsync -a $RLD/$run gpu1:$REMOTE_RLD/ 2>&1 | grep -v Ubuntu
  scp -q $log gpu1:$REMOTE_BIG/ 2>&1 | grep -v Ubuntu
  remote "echo '$job $run $it' >> $REMOTE_BIG/rl_runs.txt"
  echo "-- rl $job done: run $run iteration $it, copied to gpu1 $(date +%H:%M)"
done
echo "### stage35 local RL done $(date)"
