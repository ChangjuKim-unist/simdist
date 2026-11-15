#!/bin/bash
# Design doc 24: FiLM context (per-episode xi) on go1_paper. gpu1, 4 GPUs.
#   phase 1 training (4 x ~3.5 h) -> phase 2 gate 0 usage, SimDist fine-tunes, xi fits -> phase 3 planner evaluations.
#   Phase 3 waits for results/stage24/planner_w.txt (terminal value weight from design 23; "1.0" if it did not pass)
#   and reads results/stage24/interp_alpha.txt if design 23's interpolation passed (adds the "B + interpolation" arm).
#   usage (gpu1): tmux new -d -s s24 'bash scripts/sim2sim/stage24.sh > results/stage24_server.log 2>&1'
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage24; mkdir -p $OUT
DS=paper_sub8192; FR=sim2sim_friction0.15_go1_paper_b45000; MO=sim2sim_very_weak_motor_go1_paper_b45000
HAS() { ls -d checkpoints/models/$1/[0-9]* >/dev/null 2>&1; }
echo "### stage24 start $(date)"

# ---- phase 1: continued training ----
train() { local g=$1 name=$2 seed=$3; shift 3
  HAS $name && { echo "== $name exists"; return; }
  echo "== train $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/finetune_model.py data.dataset_name=$DS system=go1 checkpoint.resume_checkpoint=go1_paper \
    run_name=$name loss=world_model heads=all training.max_steps=60000 training.batch_size=128 training.warmup_steps=2000 \
    training.decay_steps=60000 training.end_learning_rate=0.00005 training.eval_interval=10000 training.seed=$seed "$@" 2>&1 \
    | grep -E "Added a FiLM|Per-episode|Parameters to be optimized|Steps: [0-9]*0000,|Error|Traceback" | cut -c1-400
  echo "== train $name done $(date +%H:%M)"; }
train 0 go1_paper_film_s42 42 adapt.film.enabled=true adapt.film.n_episodes=8192 adapt.film.init_seed=0 &
sleep 30; train 1 go1_paper_film_s43 43 adapt.film.enabled=true adapt.film.n_episodes=8192 adapt.film.init_seed=1 &
sleep 30; train 2 go1_paper_cont_s42 42 &
sleep 30; train 3 go1_paper_cont_s43 43 &
wait
for m in go1_paper_film_s42 go1_paper_film_s43 go1_paper_cont_s42 go1_paper_cont_s43; do HAS $m || { echo "### training $m failed"; exit 1; }; done

# ---- phase 2: gate 0 usage, SimDist fine-tunes (design 20 V1 recipe), xi fits ----
ft() { local g=$1 name=$2 base=$3 data=$4
  HAS $name && return
  echo "== finetune $name gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/finetune_model.py data.dataset_name=$data system=go1 checkpoint.resume_checkpoint=$base \
    run_name=$name training.max_steps=3000 training.eval_interval=3000 training.seed=42 \
    loss=world_model_dynamics_only heads=world_model_dynamics_only 2>&1 | grep -E "Parameters to be optimized|Error|Traceback" | cut -c1-300; }
fit() { local g=$1 film=$2 data=$3 tag=$4 cmp=$5
  echo "== xi fit $film $tag gpu$g $(date +%H:%M)"
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage24_xi_fit.py --ckpt $film --dataset $data --compare_ckpt $cmp \
    --out /workspace/simdist/$OUT/xi_${film}_$tag.json 2>&1 | grep -E "RESULT|Traceback|rror:" | cut -c1-700; }
use() { local g=$1 film=$2
  GPUS="\"device=$g\"" $RUN python scripts/sim2sim/stage24_xi_fit.py --ckpt $film --dataset $DS --usage \
    --out /workspace/simdist/$OUT/usage_$film.json 2>&1 | grep -E "RESULT|Traceback|rror:" | cut -c1-700; }
( use 0 go1_paper_film_s42; ft 0 go1_paper_cont_s42_fr_dyn go1_paper_cont_s42 $FR; ft 0 go1_paper_film_s42_fr_dyn go1_paper_film_s42 $FR ) &
( use 1 go1_paper_film_s43; ft 1 go1_paper_cont_s42_mo_dyn go1_paper_cont_s42 $MO; ft 1 go1_paper_film_s42_mo_dyn go1_paper_film_s42 $MO ) &
( ft 2 go1_paper_cont_s43_fr_dyn go1_paper_cont_s43 $FR ) &
( ft 3 go1_paper_cont_s43_mo_dyn go1_paper_cont_s43 $MO ) &
wait
GAIN=$(python3 -c "import json;print(max(json.load(open('$OUT/usage_go1_paper_film_s%d.json'%s))['rel_gain'] for s in (42,43)))")
echo "== gate 0: best relative latent-MSE gain of own xi_e over xi=0: $GAIN"
if python3 -c "import sys;sys.exit(0 if $GAIN < 0.02 else 1)"; then echo "### gate 0 stop: slot unused (< 2 %)"; exit 0; fi
( fit 0 go1_paper_film_s42 $FR fr go1_paper_cont_s42_fr_dyn ) &
( fit 1 go1_paper_film_s42 $MO mo go1_paper_cont_s42_mo_dyn ) &
( fit 2 go1_paper_film_s43 $FR fr go1_paper_cont_s43_fr_dyn ) &
( fit 3 go1_paper_film_s43 $MO mo go1_paper_cont_s43_mo_dyn ) &
wait

# ---- phase 3: planner evaluations (6 seeds x 3 speeds x 5000 steps) ----
while [ ! -f $OUT/planner_w.txt ]; do sleep 60; done
W=$(cat $OUT/planner_w.txt); ALPHA=$(cat $OUT/interp_alpha.txt 2>/dev/null || echo "")
echo "== planner value weight w = $W, interpolation alpha = ${ALPHA:-none}"
phys() { case $1 in nominal) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=1.0";;
  motor0.4) echo "sim.friction=0.8 sim.add_mass=0.0 sim.motor_strength=0.4";;
  friction0.15) echo "sim.friction=0.15 sim.add_mass=0.0 sim.motor_strength=1.0";; esac; }
Q=$OUT/queue.txt; LOCK=$OUT/queue.lock; : > $Q
X=/workspace/simdist/$OUT
for c in friction0.15 motor0.4 nominal; do
  echo "A_cont|go1_paper_cont_s42|$c|control.value_weight=$W" >> $Q
  echo "C_film_xi0|go1_paper_film_s42|$c|control.value_weight=$W" >> $Q
  for d in fr mo; do
    echo "B_cont_${d}FT|go1_paper_cont_s42_${d}_dyn|$c|control.value_weight=$W" >> $Q
    echo "D_film_xi_${d}|go1_paper_film_s42|$c|control.value_weight=$W sim.xi_json=$X/xi_go1_paper_film_s42_$d.json" >> $Q
    echo "E_film_${d}FT|go1_paper_film_s42_${d}_dyn|$c|control.value_weight=$W" >> $Q
    [ -n "$ALPHA" ] && echo "B_cont_${d}FT_interp|go1_paper_cont_s42_${d}_dyn|$c|control.value_weight=$W model.interp_from=go1_paper_cont_s42 model.interp_alpha=$ALPHA" >> $Q
    if [ "$W" != "1.0" ] && [ "$c" != nominal ]; then
      echo "B_cont_${d}FT_w1|go1_paper_cont_s42_${d}_dyn|$c|" >> $Q
      echo "D_film_xi_${d}_w1|go1_paper_film_s42|$c|sim.xi_json=$X/xi_go1_paper_film_s42_$d.json" >> $Q
    fi
  done
  echo "Apol_cont|go1_paper_cont_s42|$c|control.prior_only=true" >> $Q
  echo "Cpol_film|go1_paper_film_s42|$c|control.prior_only=true" >> $Q
done
echo "B_cont_frFT_s43|go1_paper_cont_s43_fr_dyn|friction0.15|control.value_weight=$W" >> $Q
echo "B_cont_moFT_s43|go1_paper_cont_s43_mo_dyn|motor0.4|control.value_weight=$W" >> $Q
echo "D_film_xi_fr_s43|go1_paper_film_s43|friction0.15|control.value_weight=$W sim.xi_json=$X/xi_go1_paper_film_s43_fr.json" >> $Q
echo "D_film_xi_mo_s43|go1_paper_film_s43|motor0.4|control.value_weight=$W sim.xi_json=$X/xi_go1_paper_film_s43_mo.json" >> $Q
echo "== $(wc -l < $Q) evaluation jobs"
worker() { local g=$1
  while true; do
    local job; job=$(flock $LOCK bash -c "head -1 $Q; sed -i 1d $Q")
    [ -z "$job" ] && break
    IFS='|' read -r name ckpt cond extra <<< "$job"
    echo "== $name $cond gpu$g $(date +%H:%M)"
    GPUS="\"device=$g\"" $RUN python scripts/simulate_batch.py model.checkpoint=$ckpt $(phys $cond) sim.terrain=plane sim.total_steps=5000 \
      sim.episode_length_s=100 'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=$X/eval_gpu$g.csv \
      output.model_name=$name output.condition=$cond $extra --headless 2>&1 | grep -E "Traceback|rror:" | head -3
  done; }
for g in 0 1 2 3; do worker $g & sleep 20; done
wait
echo "### stage24 done $(date)"
