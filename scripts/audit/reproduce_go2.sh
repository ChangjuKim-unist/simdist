#!/bin/bash
# Reproduction of the SimDist paper's quadruped simulation ablation (Table I, "SimDist" = 22.78) with the
# authors' released Go2 checkpoint (checkpoints/models/go2_2026-02-12, upstream commit c0d9eac) on our evaluator
# (scripts/simulate_batch.py). Paper protocol (Appendix D): every combination of commanded speed, ground friction,
# terrain type and terrain difficulty, one episode each, 1000 steps or failure, metric = accumulated reward per episode.
# The paper's Table X values are only partly legible (speeds/frictions 0.2 .. 1.2, difficulties 0.2 .. 1.0,
# six terrains -> 6 x 6 x 6 x 5 = 1080); GRID=full runs that grid, GRID=reduced a 3 x 3 subset of friction and difficulty.
# Usage: GRID=reduced nohup bash scripts/audit/reproduce_go2.sh > results/audit/repro_go2.log 2>&1 &
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh; export SIMDIST_DIR=~/simdist
OUT=results/audit/repro_go2.csv
TERRAINS="boxes random_rough pyramid_stairs pyramid_stairs_inv hf_pyramid_slope hf_pyramid_slope_inv"
case ${GRID:-reduced} in
  full) FRICTIONS="0.2 0.4 0.6 0.8 1.0 1.2"; DIFFS="0.2 0.4 0.6 0.8 1.0";;
  *) FRICTIONS="0.2 0.7 1.2"; DIFFS="0.2 0.6 1.0";;
esac
echo "### repro start $(date) grid=${GRID:-reduced}"
for t in $TERRAINS; do for f in $FRICTIONS; do for d in $DIFFS; do
  cond="${t}_f${f}_d${d}"
  grep -q ",$cond," $OUT 2>/dev/null && continue
  echo "== $cond $(date +%H:%M)"
  $RUN python scripts/simulate_batch.py model.checkpoint=go2_2026-02-12 sim.terrain=$t sim.friction=$f sim.terrain_difficulty=$d \
    sim.total_steps=1000 'batch.speeds=[0.2,0.4,0.6,0.8,1.0,1.2]' 'batch.seeds=[42]' \
    output.csv=/workspace/simdist/$OUT output.model_name=go2_authors output.condition=$cond --headless 2>&1 \
    | grep -E "RESULT|Traceback|Error:" | tail -7
done; done; done
echo "### repro end $(date)"
