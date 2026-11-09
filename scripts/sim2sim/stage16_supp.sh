#!/bin/bash
# Design doc 16 §7: supplementary runs on gpu1 (queue over GPUS_LIST).  Needs results/stage15/gen_broad.npz, gen_paper.npz,
# results/stage14/head_*.npz on the server.  Every run also writes per-term rewards (output.terms_csv).
cd ~/simdist; RUN=~/simdist-docker/run.sh; OUT=results/stage16; mkdir -p $OUT $OUT/traces
GPUS_LIST=${GPUS_LIST:-"0 1 2 3"}
phys() { local m=1.0 f=0.8 a=0.0 tok; IFS=_ read -ra toks <<< "$1"
  for tok in "${toks[@]}"; do case $tok in motor*) m=${tok#motor};; friction*) f=${tok#friction};; mass*) a=${tok#mass};; esac; done
  echo "sim.friction=$f sim.add_mass=$a sim.motor_strength=$m"; }
GEN="control.gen.enabled=true control.gen.weights=/workspace/simdist/results/stage15/gen_broad.npz"
GENP="control.gen.enabled=true control.gen.weights=/workspace/simdist/results/stage15/gen_paper.npz"
PO="control.prior_only=true"
B0="control.init_mode=prior control.mixture_coef=0.0"
cal() { echo "control.calib.enabled=true control.calib.weights=/workspace/simdist/results/stage14/$1.npz control.calib.tau=0.5"; }
run() { local gpu=$1 name=$2 model=$3 cond=$4; shift 4
  local csv=$OUT/eval_supp_gpu$gpu.csv
  if cat $OUT/eval_supp*.csv 2>/dev/null | grep -q "^$name,$cond,"; then return; fi
  local extra=""; [ "$1" = trace ] && { extra="output.trace_npz=/workspace/simdist/$OUT/traces/${name}_$cond.npz"; shift; }
  echo "== $name ($model) on $cond ($*) gpu$gpu $(date +%H:%M)"
  GPUS="\"device=$gpu\"" $RUN python scripts/simulate_batch.py model.checkpoint=$model $(phys $cond) sim.terrain=plane sim.total_steps=5000 sim.episode_length_s=100 \
    'batch.speeds=[0.3,0.6,0.9]' 'batch.seeds=[11,12,13,14,15,16]' output.csv=/workspace/simdist/$csv output.terms_csv=/workspace/simdist/$OUT/terms_supp_gpu$gpu.csv \
    output.model_name=$name output.condition=$cond $extra "$@" --headless 2>&1 | grep -E "Traceback|rror:"
}
QUEUE=$OUT/queue_supp.txt; LOCK=$OUT/queue_supp.lock
cat > $QUEUE <<Q
S3_gen_q50 go1_broadwm friction0.15 trace $GEN $(cal head_broadwm_noF)
S1c_jhat_std0.5 go1_broadwm friction0.15 control.init_std=0.5
S1d_corr_std0.1_q50 go1_broadwm friction0.15 $B0 control.init_std=0.1 control.noise_corr=0.9 $(cal head_broadwm_noF)
S1d_corr_std0.3_q50 go1_broadwm friction0.15 $B0 control.init_std=0.3 control.noise_corr=0.9 $(cal head_broadwm_noF)
S1c_jhat_std1.0 go1_broadwm friction0.15 control.init_std=1.0
S1c_q50_std0.5 go1_broadwm friction0.15 control.init_std=0.5 $(cal head_broadwm_noF)
S1d_corr_std0.3_q50_i16 go1_broadwm friction0.15 $B0 control.init_std=0.3 control.noise_corr=0.9 $(cal head_broadwm_noF) control.iterations=16
S2_gen_q50 go1_broadwm motor0.3 $GEN $(cal head_broadwm_noM)
S2_gen_jhat go1_broadwm motor0.3 $GEN
S2_gen_q50 go1_broadwm motor0.5_friction0.3 $GEN $(cal head_broadwm_noM)
S2_gen_q50 go1_broadwm motor0.7_mass3 $GEN $(cal head_broadwm_noM)
S2_gen_q50 go1_broadwm friction0.3_mass3 $GEN $(cal head_broadwm_noF)
S2_gen_q50 go1_broadwm friction0.3 $GEN $(cal head_broadwm_noF)
S2_gen_q50 go1_broadwm friction0.5 $GEN $(cal head_broadwm_noF)
S2_gen_q50 go1_broadwm mass1 $GEN $(cal head_broadwm_noA)
S2_gen_q50 go1_broadwm mass5 $GEN $(cal head_broadwm_noA)
S2_gen_q50 go1_broadwm motor0.6 $GEN $(cal head_broadwm_noM)
S2_gen_q50 go1_broadwm motor0.8 $GEN $(cal head_broadwm_noM)
S1a_genp_q50 go1_paper friction0.15 $GENP $(cal head_paper2_noF)
S1a_genp_jhat go1_paper friction0.15 $GENP
S1a_genp_q50 go1_paper motor0.4 $GENP $(cal head_paper2_noM)
S1a_genp_jhat go1_paper motor0.4 $GENP
S1b_paperfr_genp_q50 go1_paper_friction0.15_dyn_b45000_term friction0.15 $GENP $(cal head_paper2_noF)
S1b_paperfr_genp_jhat go1_paper_friction0.15_dyn_b45000_term friction0.15 $GENP
S1b_papermotor_genp_q50 go1_paper_very_weak_motor_dyn_b45000_term motor0.4 $GENP $(cal head_paper2_noM)
S3_prior go1_broadwm nominal $PO
S3_prior go1_broadwm friction0.15 $PO
S3_jhat go1_broadwm nominal
S3_jhat go1_broadwm friction0.15
S3_gen_q50 go1_broadwm nominal $GEN $(cal head_broadwm_noF)
Q
pop() { ( flock 9; local j; j=$(head -1 $QUEUE); [ -n "$j" ] && sed -i '1d' $QUEUE; echo "$j" ) 9>$LOCK; }
worker() { local gpu=$1; while :; do local j; j=$(pop); [ -z "$j" ] && break; run $gpu $j; done; echo "== worker gpu$gpu idle $(date +%H:%M)"; }
for g in $GPUS_LIST; do worker $g & sleep 20; done
wait
python3 - <<'PY'
import csv, glob
for stem in ('eval_supp', 'terms_supp'):
    rows=[]
    for f in sorted(glob.glob(f'results/stage16/{stem}_gpu*.csv')): rows+=list(csv.DictReader(open(f)))
    if rows:
        with open(f'results/stage16/{stem}.csv','w',newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print('merged', stem, len(rows), 'rows')
PY
echo "### supp eval done $(date)"
