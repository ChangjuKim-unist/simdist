#!/usr/bin/env bash
# Design doc 6: motor-strength DR width sweep. Trains Go1DR experts with narrower ranges than the
# broad expert (0.3-1.1, run 2026-10-20_07-40-06); the nominal expert (no motor DR) is 2026-10-01_10-38-36.
# Each run ~6 h on the 3060 (5000 PPO iterations). Run ids are recorded in results/stage1/experts.txt.
set -u; cd ~/simdist; RUN=~/simdist-docker/run.sh
mkdir -p results/stage1; OUT=results/stage1/experts.txt
for range in "0.5,1.1" "0.7,1.1" "0.9,1.1"; do
  if grep -q "^$range " $OUT 2>/dev/null; then continue; fi
  echo "== expert motor DR $range $(date)"
  before=$(ls -t checkpoints/rl | head -1)
  $RUN env GO1_MOTOR_DR=$range python scripts/train_rl.py --task Go1DR --headless 2>&1 | grep -E "Traceback|rror:|GO1_MOTOR_DR|Learning iteration 4999" | head -5
  after=$(ls -t checkpoints/rl | head -1)
  if [ "$after" != "$before" ] && [ -f checkpoints/rl/$after/model_4999.pt ]; then echo "$range $after" >> $OUT; echo "   -> $after"; else echo "   !! run for $range did not finish (latest dir $after)"; fi
done
echo "### width sweep experts done $(date)"
