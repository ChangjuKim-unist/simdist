#!/usr/bin/env bash
# Design doc 11 S2b: identification accuracy with 5-minute (15k-step) logs per true scale, both collecting policies,
# more replicas/rounds. Runs after stage 7.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; P=scripts/sim2sim/run_pipeline.sh; OUT=results/stage7; mkdir -p $OUT
until grep -q "### stage7 done" results/stage7_strengthen2.log 2>/dev/null; do sleep 300; done
echo "### idcurve start $(date)"
for s in 0.3 0.5 0.6 0.8; do for spec in "go1_paper_broadhead:bh" "go1_paper:sd"; do IFS=: read ck tag <<< "$spec"
  ds=sim2sim_motor${s}_${tag}_1min   # the 1-round collection = 15k steps (3 speeds x 5000)
  [ -d datasets/real/${ds}_b15000/processed_data_go1_H-25_T-25 ] || COND=motor$s DATASET=$ds BUDGETS=15000 $P process 2>&1 | grep -E "Error|Traceback" | head -2
  [ -f $OUT/id_s2b_${tag}_s$s.json ] || { echo "== identify $tag true $s (15k) $(date +%H:%M)"; $RUN python scripts/sim2sim/identify_physics.py dataset=${ds}_b15000 out=$OUT/id_s2b_${tag}_s$s.json replicas=24 rounds=100 --headless 2>&1 | grep -E "MAP|Error|Traceback" | cut -c1-160; }
done; done
echo "### idcurve done $(date)"
