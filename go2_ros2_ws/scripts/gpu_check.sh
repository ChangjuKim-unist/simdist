#!/usr/bin/env bash
# Refuse to start the MPPI controller while another process (a simulation
# experiment on the host) already occupies the GPU: the planner then runs at
# 25-45 Hz instead of 50 Hz and the robot falls. Override with GPU_CHECK=0.
[ "${GPU_CHECK:-1}" = "0" ] && exit 0
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -z "$used" ] && { echo "gpu_check: nvidia-smi unavailable, continuing"; exit 0; }
if [ "$used" -gt 1500 ]; then
  echo "!!! GPU already in use (${used} MiB). Stop the simulation experiment on the host first"
  echo "!!! (docker ps / pkill -f run_pipeline.sh), or run with GPU_CHECK=0 to override."
  exit 1
fi
echo "gpu_check: GPU free (${used} MiB used)"
