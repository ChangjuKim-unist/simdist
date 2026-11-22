#!/bin/bash
# Design 40 §9 level screen (PPO only). Run from ~/simdist on gpu1. Arg: GPU index.
G=${1:-0}; D=results/stage40; PPO=/mnt/newssd/cjkim/toy/runs/ppo_s0/params.pkl
L="mot40 mot30 mot25 legR30 legR20 legR10 delay1 delay2 delay3 tow20 tow30 tow40"
CMD="cd /workspace/simdist && export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/newssd/cjkim/toy_pylibs:scripts/toy;"
for c in $L; do CMD="$CMD python scripts/toy/toy_ppo_eval.py --params $PPO --cond $c --out $D/screen_ppo_$c;"; done
CMD="$CMD python scripts/toy/stage40_screen_report.py --d $D --md $D/screen.md"
GPUS="\"device=$G\"" ~/simdist-docker/run.sh bash -c "$CMD"
