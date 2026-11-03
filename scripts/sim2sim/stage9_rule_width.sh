#!/usr/bin/env bash
# Design doc 12: DR width as budget allocation. E1 width-cost curve (fit k), E4 reference evals at motor 0.5/0.6,
# E2 rule-width heads for true 0.5/0.6, E3 friction-blanket head (unidentifiable axis control).
# Heads reuse the stage6 pipeline: PPO expert (Go1DR) -> export -> data (384 envs x 20k) -> process -> policy-only
# head on go1_paper (60k steps). All sim2sim evals: 18 robots x 100 s, speeds 0.3/0.6/0.9, seeds 11-16.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage9; mkdir -p $OUT; CSV=$OUT/eval_rule_width.csv
echo "### rule-width start $(date)"

# ---- head pipeline: head NAME MOTOR_LO,MOTOR_HI [FRICTION_LO,FRICTION_HI] ----
mkhead() { local name=$1 mdr=$2 fdr=${3:-}; local env="GO1_MOTOR_DR=$mdr"; [ -n "$fdr" ] && env="$env GO1_FRICTION_DR=$fdr"
  local EXP=$OUT/expert_$name.txt DSF=$OUT/dataset_$name.txt
  if [ ! -f $EXP ]; then
    local before=$(ls -t checkpoints/rl | head -1); echo "== expert $name ($env) $(date +%H:%M)"
    $RUN env $env python scripts/train_rl.py --task Go1DR --headless --num_envs 1024 2>&1 | grep -E "Traceback|rror:|Learning iteration 4999" | head -3
    local after=$(ls -t checkpoints/rl | head -1)
    [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ] && echo "$after" > $EXP || { echo "expert $name failed"; return 1; }
  fi
  local RL=$(cat $EXP); echo "== expert $name = $RL"
  [ -f checkpoints/rl/$RL/policies/policy_4999.pt ] || $RUN python scripts/export_policies.py --task Go1DRPlay -r $RL --headless 2>&1 | grep -E "Error|Traceback" | tail -2
  if [ ! -f $DSF ]; then
    local before=$(ls datasets/sim | sort | tail -1); echo "== data $name ($env) $(date +%H:%M)"
    $RUN env $env python scripts/generate_data.py rl_run=$RL system=go1dr envs=384 steps=20000 2>&1 | grep -E "Error|Traceback" | head -2
    local ds=$(ls datasets/sim | sort | tail -1); [ "$ds" != "$before" ] || { echo "no dataset for $name"; return 1; }; echo "$ds" > $DSF
    $RUN python scripts/process_data.py dataset_name=$ds system=go1dr 2>&1 | grep -E "Error|Traceback" | head -2
  fi
  local DS=$(cat $DSF)
  ls -d checkpoints/models/go1_paper_$name/[0-9]* >/dev/null 2>&1 && return 0
  echo "== head $name $(date +%H:%M)"
  $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1dr checkpoint.resume_checkpoint=go1_paper run_name=go1_paper_$name loss=world_model heads=policy_only training.max_steps=60000 training.eval_interval=10000 2>&1 | grep -E "Steps: |Error|Traceback" | cut -c1-160 | tail -2
  ls -d checkpoints/models/go1_paper_$name/[0-9]* >/dev/null 2>&1 || { echo "head $name failed"; return 1; }
}

phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  motor0.5) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.5";;
  motor0.6) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.6";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
SEEDS="[11,12,13,14,15,16]"
run() { local name=$1 ckpt=$2 cond=$3; shift 3
  if [ -f $CSV ] && grep -q "^$name,$cond," $CSV; then return; fi
  echo "== $name on $cond $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' "batch.seeds=$SEEDS" output.csv=/workspace/simdist/$CSV output.model_name=$name output.condition=$cond "$@" --headless 2>&1 | grep -E "Traceback|rror:" | head -3
}

# ---- E1: width-cost curve (lower bound 0.3 fixed, upper varies; eval at motor 0.4 inside all) ----
mkhead rw_w02 0.3,0.5 && for c in motor0.4 nominal; do run H_rw_w02_planner go1_paper_rw_w02 $c; done
mkhead rw_w06 0.3,0.9 && for c in motor0.4 nominal; do run H_rw_w06_planner go1_paper_rw_w06 $c; done

# (E4 reference evals at motor 0.5/0.6 live in stage10_reeval.sh, results/stage10/eval_core.csv)

# ---- fit k: falls(w) = a * w^k on (0.2, 0.4, 0.6, 0.8); fallback k=1.8 if non-monotone or slope < 0.5 ----
# falls = num_falls (base/thigh contact or bad orientation) per 1000 steps; all four points from the fixed evaluator
python3 - <<'EOF' > $OUT/k.json
import csv, json, math, numpy as np
def falls(path, model, cond):
    v=[float(r['num_falls']) for r in csv.DictReader(open(path)) if r['model']==model and r['condition']==cond]
    return sum(v)/len(v)/5 if v else None
pts={0.2:falls('results/stage9/eval_rule_width.csv','H_rw_w02_planner','motor0.4'),
     0.4:falls('results/stage10/eval_core.csv','H_targethead_planner','motor0.4'),
     0.6:falls('results/stage9/eval_rule_width.csv','H_rw_w06_planner','motor0.4'),
     0.8:falls('results/stage10/eval_core.csv','H_broadhead_planner','motor0.4')}
ok={w:f for w,f in pts.items() if f}
k=None; note='fit'
if len(ok)>=3:
    w=np.array(sorted(ok)); f=np.array([ok[x] for x in w]); f=np.maximum(f,0.05)
    k=float(np.polyfit(np.log(w),np.log(f),1)[0])
    if not np.all(np.diff(f)>=0) or k<0.5: note=f'fallback (fit k={k:.2f}, monotone={bool(np.all(np.diff(f)>=0))})'; k=1.8
else: note='fallback (missing points)'; k=1.8
json.dump({'points':pts,'k':k,'note':note},open('/dev/stdout','w'))
EOF
K=$(python3 -c "import json;print(round(json.load(open('$OUT/k.json'))['k'],2))"); echo "== k = $K ($(cat $OUT/k.json))"

# ---- E2: rule-width heads for true 0.5 and 0.6 (posteriors from the re-identified broadhead logs, results/stage10;
#      bias set = MAP - true of the OTHER calibration points {0.5, 0.6, 0.8}, leave-one-out) ----
for s in 0.5 0.6; do
  tag=${s/./}
  BIAS=$(python3 -c "
import json
b=[]
for t in [0.5,0.6,0.8]:
    if abs(t-$s)<1e-6: continue
    try: b.append(round(json.load(open(f'results/stage10/id_bh_s{t}.json'))['map'][0]-t,3))
    except Exception: pass
print(' '.join(str(x) for x in b) if b else '0.0')")
  R=$(python3 scripts/sim2sim/dr_from_posterior.py results/stage10/id_bh_s$s.json --k $K --bias $BIAS 2>$OUT/rule_t$tag.json); echo "== rule range true $s: $R (k=$K, bias=$BIAS)"
  mkhead rw_t$tag $R && for c in motor$s nominal; do run H_rw_t${tag}_planner go1_paper_rw_t$tag $c; done
done

# ---- E3: friction-blanket head (no identification; lower bound 0.1 covers friction 0.15) ----
mkhead fr_blanket 1.0,1.0 0.1,1.2 && for c in friction0.15 nominal; do run H_fr_blanket_planner go1_paper_fr_blanket $c; done
echo "### rule-width done $(date)"
