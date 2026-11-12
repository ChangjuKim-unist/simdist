#!/bin/bash
# Design 18 section 8 (docs/design_branch_headroom_ko.md): supplement before gate 1, on gpu1 (free GPUs only).
#   check 2: branch runs on the gate-0 anchors with candidates A0, A1, D, F_cl, C_cl (+ reward-term sums), 8 cells
#   check 3: anchors + branch runs with the design-16 generator mean as the base controller, 6 conditions,
#            candidates A0, D, E, C (open loop), G (closed loop)
# Results: $BIG (spare SSD, symlinked from results/stage18). Re-runnable (finished runs are skipped, branch runs resume).
# Usage:  GPUS_LIST="0 1 2 3" nohup bash scripts/sim2sim/stage18_supp.sh > results/stage18_supp.log 2>&1 &
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage18; mkdir -p $BIG; [ -e results/stage18 ] || ln -s $BIG results/stage18; OUT=results/stage18
export SIMDIST_DIR=~/simdist
GPUS_LIST=${GPUS_LIST:-"0 1 2 3"}
W2=${W2:-8}     # branch points per round, check 2 (F_cl planner over W2 x J robots)
W3=${W3:-8}     # check 3 (25 candidates x J = 75 robots per point)
GEN=/workspace/simdist/results/stage15/gen_broad.npz
echo "### stage18 supp start $(date) GPUs: $GPUS_LIST W2=$W2 W3=$W3"

phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
COMMON="control.prior_only=true sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100"
mark_done() {  # $1 anchor npz, $2 branch csv, $3 done file
  python3 - "$1" "$2" "$3" <<'PY'
import sys, numpy as np, csv
a = np.load(sys.argv[1]); keys = {(float(v), int(s), int(t)) for v, s, t in zip(a["speed"], a["seed"], a["step"])}
done = set()
try:
    done = {(float(r["speed"]), int(r["seed"]), int(r["step"])) for r in csv.DictReader(open(sys.argv[2]))}
except FileNotFoundError:
    pass
if keys and keys <= done:
    open(sys.argv[3], "w").write("done\n"); print(f"branch complete: {len(keys)} points")
else:
    print(f"branch incomplete: {len(done & keys)}/{len(keys)} points")
PY
}
branch2() { local gpu=$1 name=$2 model=$3 cond=$4
  [ -f $OUT/branch2_${name}_$cond.done ] && return
  echo "== branch2 $name ($model) on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$model $(phys $cond) $COMMON \
    branch.anchor=/workspace/simdist/$OUT/anchor_${name}_$cond.npz branch.csv=/workspace/simdist/$OUT/branch2_${name}_$cond.csv branch.W=$W2 \
    'branch.groups=[A0,A1,D,F,Fcl,Ccl]' branch.terms=true branch.gen_weights=$GEN \
    output.model_name=$name output.condition=$cond --headless 2>&1 | grep -E "RESULT branch|Traceback|rror:|round [0-9]*0/" | tail -4
  mark_done $OUT/anchor_${name}_$cond.npz $OUT/branch2_${name}_$cond.csv $OUT/branch2_${name}_$cond.done
}
anchor3() { local gpu=$1 cond=$2
  [ -f $OUT/anchor_gen_$cond.npz ] && return
  echo "== anchor3 gen on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=anchor model.checkpoint=go1_broadwm $(phys $cond) $COMMON \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' branch.base=gen branch.shadow=false branch.gen_weights=$GEN \
    branch.out=/workspace/simdist/$OUT/anchor_gen_$cond.npz \
    output.csv=/workspace/simdist/$OUT/anchor_runs_gpu$gpu.csv output.model_name=gen output.condition=$cond --headless 2>&1 \
    | grep -E "RESULT|branch points written|Traceback|rror:" | tail -4
}
branch3() { local gpu=$1 cond=$2
  [ -f $OUT/branch3_gen_$cond.done ] && return
  echo "== branch3 gen on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=go1_broadwm $(phys $cond) $COMMON \
    branch.anchor=/workspace/simdist/$OUT/anchor_gen_$cond.npz branch.csv=/workspace/simdist/$OUT/branch3_gen_$cond.csv branch.W=$W3 \
    'branch.groups=[A0,D,E,C,G]' branch.base=gen branch.terms=true branch.gen_weights=$GEN \
    output.model_name=gen output.condition=$cond --headless 2>&1 | grep -E "RESULT branch|Traceback|rror:|round [0-9]*0/" | tail -4
  mark_done $OUT/anchor_gen_$cond.npz $OUT/branch3_gen_$cond.csv $OUT/branch3_gen_$cond.done
}

QUEUE=$OUT/queue_supp.txt; LOCK=$OUT/queue_supp.lock
RUNS2="broadwm:go1_broadwm:nominal broadwm:go1_broadwm:motor0.4 broadwm:go1_broadwm:friction0.15 paper:go1_paper:friction0.15
paperfr:go1_paper_friction0.15_dyn_b45000_term:friction0.15 paperfr:go1_paper_friction0.15_dyn_b45000_term:motor0.4
papermotor:go1_paper_very_weak_motor_dyn_b45000_term:friction0.15 papermotor:go1_paper_very_weak_motor_dyn_b45000_term:motor0.4"
COND3="nominal motor0.4 friction0.15 mass3 motor0.35 friction0.1"
{ for c in $COND3; do echo "ANCHOR3 $c"; done
  for r in $RUNS2; do IFS=: read -r n m c <<< "$r"; echo "BRANCH2 $n $m $c"; done
  for c in $COND3; do echo "BRANCH3 $c"; done; } > $QUEUE
echo "== queue: $(wc -l < $QUEUE) jobs"
ready() { set -- $1; case $1 in ANCHOR3) return 0;; BRANCH2) [ -f $OUT/anchor_$2_$4.npz ];; BRANCH3) [ -f $OUT/anchor_gen_$2.npz ];; esac; }
pop() { ( flock 9; local line
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    if ready "$line"; then grep -v -x -F -- "$line" $QUEUE > $QUEUE.tmp; mv $QUEUE.tmp $QUEUE; echo "$line"; return; fi
  done < $QUEUE; echo "" ) 9>$LOCK; }
worker() { local gpu=$1 j idle=0
  while :; do
    j=$(pop)
    if [ -z "$j" ]; then
      [ -s $QUEUE ] || break
      if ! compgen -G "$OUT/running_supp_gpu*" >/dev/null; then idle=$((idle+1)); [ $idle -ge 10 ] && { echo "== worker gpu$gpu: queue blocked, giving up $(date +%H:%M)"; break; }; else idle=0; fi
      sleep 60; continue
    fi
    idle=0; echo "$j" > $OUT/running_supp_gpu$gpu
    set -- $j
    case $1 in ANCHOR3) anchor3 $gpu $2;; BRANCH2) branch2 $gpu $2 $3 $4;; BRANCH3) branch3 $gpu $2;; esac
    rm -f $OUT/running_supp_gpu$gpu
  done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
echo "### stage18 supp end $(date)"
