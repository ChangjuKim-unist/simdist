#!/bin/bash
# Design 18 gate 0 (docs/design_branch_headroom_ko.md): counterfactual-branching headroom measurement on gpu1 (4 GPUs).
# Anchor runs (base controller + shadow planner, snapshots) then branch runs (all candidate groups, J=3) per model x condition.
# Results on the spare SSD (root disk is nearly full): $OUT -> /mnt/newssd/cjkim/simdist/results/stage18 (symlinked from results/stage18).
# Usage:  nohup bash scripts/sim2sim/stage18_gate0.sh > results/stage18_gate0.log 2>&1 &
# Re-runnable: finished anchors (npz) / branch runs (.done) are skipped; a branch run resumes from its CSV.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
BIG=/mnt/newssd/cjkim/simdist/results/stage18; mkdir -p $BIG; [ -e results/stage18 ] || ln -s $BIG results/stage18; OUT=results/stage18
export SIMDIST_DIR=~/simdist
GPUS_LIST=${GPUS_LIST:-"0 1 2 3"}
W=${W:-8}
echo "### stage18 gate0 start $(date) GPUs: $GPUS_LIST"

phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
COMMON="control.prior_only=true sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100"
anchor() { local gpu=$1 name=$2 model=$3 cond=$4; shift 4
  [ -f $OUT/anchor_${name}_$cond.npz ] && return
  echo "== anchor $name ($model) on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=anchor model.checkpoint=$model $(phys $cond) $COMMON \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' branch.out=/workspace/simdist/$OUT/anchor_${name}_$cond.npz \
    output.csv=/workspace/simdist/$OUT/anchor_runs_gpu$gpu.csv output.model_name=$name output.condition=$cond "$@" --headless 2>&1 \
    | grep -E "branch points written|Traceback|rror:" | tail -3
}
branch() { local gpu=$1 name=$2 model=$3 cond=$4; shift 4
  [ -f $OUT/branch_${name}_$cond.done ] && return
  echo "== branch $name ($model) on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$model $(phys $cond) $COMMON \
    branch.anchor=/workspace/simdist/$OUT/anchor_${name}_$cond.npz branch.csv=/workspace/simdist/$OUT/branch_${name}_$cond.csv branch.W=$W \
    output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "RESULT branch|Traceback|rror:|round [0-9]*0/" | tail -4
  grep -q "RESULT branch" <<< "$(tail -c 2000 $OUT/branch_${name}_$cond.csv 2>/dev/null; echo)" || true
  # done when every selected branch point has rows: the script prints "0 to run" on a restart; mark done if the last run finished
  python3 - "$OUT/anchor_${name}_$cond.npz" "$OUT/branch_${name}_$cond.csv" "$OUT/branch_${name}_$cond.done" <<'PY'
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

# ---- fidelity check (design 18 section 2.5): two anchors, A0/A1, three unperturbed clones ----
fidelity() { local gpu=$1 name=$2 model=$3 cond=$4
  [ -f $OUT/fidelity_${name}_$cond.csv ] && return
  echo "== fidelity $name on $cond gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/sim2sim/branch_eval.py mode=branch model.checkpoint=$model $(phys $cond) $COMMON \
    branch.anchor=/workspace/simdist/$OUT/anchor_${name}_$cond.npz branch.csv=/workspace/simdist/$OUT/fidelity_${name}_$cond.csv \
    'branch.groups=[A0,A1]' branch.perturb=false 'branch.speeds=[0.6]' 'branch.seeds=[11,12]' branch.W=$W \
    output.model_name=$name output.condition=$cond --headless 2>&1 | grep -E "RESULT branch|Traceback|rror:" | tail -3
}

# ---- job queue (name model condition) ----
QUEUE=$OUT/queue.txt; LOCK=$OUT/queue.lock
RUNS="broadwm:go1_broadwm:nominal broadwm:go1_broadwm:motor0.4 broadwm:go1_broadwm:friction0.15 broadwm:go1_broadwm:mass3
paper:go1_paper:nominal paper:go1_paper:motor0.4 paper:go1_paper:friction0.15
paperfr:go1_paper_friction0.15_dyn_b45000_term:friction0.15 paperfr:go1_paper_friction0.15_dyn_b45000_term:motor0.4
papermotor:go1_paper_very_weak_motor_dyn_b45000_term:friction0.15 papermotor:go1_paper_very_weak_motor_dyn_b45000_term:motor0.4"
{ for r in $RUNS; do IFS=: read -r n m c <<< "$r"; echo "ANCHOR $n $m $c"; done
  echo "FIDELITY paper go1_paper nominal"; echo "FIDELITY broadwm go1_broadwm friction0.15"
  for r in $RUNS; do IFS=: read -r n m c <<< "$r"; echo "BRANCH $n $m $c"; done; } > $QUEUE
echo "== queue: $(wc -l < $QUEUE) jobs"
ready() { set -- $1; case $1 in ANCHOR) return 0;; FIDELITY|BRANCH) [ -f $OUT/anchor_$2_$4.npz ];; esac; }
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
      if ! compgen -G "$OUT/running_gpu*" >/dev/null; then idle=$((idle+1)); [ $idle -ge 10 ] && { echo "== worker gpu$gpu: queue blocked, giving up $(date +%H:%M)"; break; }; else idle=0; fi
      sleep 60; continue
    fi
    idle=0; echo "$j" > $OUT/running_gpu$gpu
    set -- $j
    case $1 in ANCHOR) anchor $gpu $2 $3 $4;; FIDELITY) fidelity $gpu $2 $3 $4;; BRANCH) branch $gpu $2 $3 $4;; esac
    rm -f $OUT/running_gpu$gpu
  done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
echo "### stage18 gate0 end $(date)"
