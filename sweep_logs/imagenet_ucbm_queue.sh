#!/bin/bash
# UCBM ImageNet: once discover_concepts.py has written concepts_3000_64, run the gate sweep (one process
# per lambda_gate, matched random control) and the native-sparsity (elastic-net lam_w) sweep, MAX_JOBS at a
# time, round-robin over GPUS. Recipe: the UCBM README's ImageNet train_cbm.py command.
cd /home/hy381/project/prelim/ucbm
eval "$(conda shell.bash hook)"; conda activate ucbm
DISCOVER_PID=${DISCOVER_PID:?pid of discover_concepts.py}
LOG=/home/hy381/project/logs
MAX_JOBS=${MAX_JOBS:-2}
GPUS=(${GPUS:-7 6 5 4 3 2 1 0})
COMMON="-d imagenet -b resnet50_v2 -c concepts_3000_64 --epochs 20 --dropout_p 0.1 --lr 0.001 --scale_choose no --bias_choose learn --normalize_concepts --relu no --k 42 --seeds 0 10 20 --save_classifiers"

while kill -0 "$DISCOVER_PID" 2>/dev/null; do sleep 300; done
ls save/RESULTS/imagenet-resnet50_v2/concept_data | grep -q concepts_3000_64 || { echo "no concepts_3000_64"; exit 1; }

jobs_list=()
for l in 1e-05 3e-05 0.0001 0.0003 0.001 0.003 0.01; do
  jobs_list+=("gated_$l|$COMMON --lam_w 1e-4 -g --lambda_gates $l --gate_forward hard --lam_pi 0 --random_control")
done
for l in 0.0 1e-05 0.0001 0.0003 0.001 0.003 0.01; do
  jobs_list+=("sparse_$l|$COMMON --lam_ws $l")
done

i=0
for job in "${jobs_list[@]}"; do
  name=${job%%|*}; args=${job#*|}
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 120; done
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  echo "$(date) start $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u train_cbm_seed_sweep.py $args > $LOG/ucbm_imagenet_$name.log 2>&1 &
done
wait
echo "$(date) all UCBM ImageNet runs done"
