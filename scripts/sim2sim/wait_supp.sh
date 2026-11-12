#!/bin/bash
# poll gpu1 for the end of the stage18 supplement driver
for i in $(seq 1 60); do
  out=$(ssh -o ConnectTimeout=15 gpu1 'tail -3 ~/simdist/results/stage18_supp.log 2>/dev/null; ls /mnt/newssd/cjkim/simdist/results/stage18/*.done 2>/dev/null | wc -l' 2>/dev/null | grep -v "^Ubuntu")
  echo "$(date +%H:%M) $out" | tail -2
  grep -q "stage18 supp end" <<< "$out" && exit 0
  sleep 600
done
exit 1
