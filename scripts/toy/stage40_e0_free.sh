#!/bin/bash
# Design 40 E0: M-free main cells, independent of the clock weight. Args: GPU cond [cond ...]
G=$1; shift; cd ~/simdist; D=results/stage40
for c in "$@"; do
  GPUS="\"device=$G\"" ~/simdist-docker/run.sh bash -c "cd /workspace/simdist && XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy python scripts/toy/toy_mppi.py --cond $c --arm free --out $D/e0_free_$c" >> $D/log_free_g$G.txt 2>&1
done
