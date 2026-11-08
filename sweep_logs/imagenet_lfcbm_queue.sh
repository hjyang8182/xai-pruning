#!/bin/bash
# LF-CBM ImageNet: once the shared stage (seed_sweep_results/shared_imagenet) exists, run the gate sweep
# (one process per lambda_gate, matched random control) and the native-sparsity sweep (one per lam),
# MAX_JOBS at a time (each holds the ~20 GB train_c), round-robin over GPUS.
cd /home/hy381/project/prelim/Label-free-CBM
eval "$(conda shell.bash hook)"; conda activate lfcbm
SHARED=seed_sweep_results/shared_imagenet
LOG=/home/hy381/project/logs
MAX_JOBS=${MAX_JOBS:-2}
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
COMMON="--dataset imagenet --backbone resnet50 --concept_set data/concept_sets/imagenet_filtered.txt --clip_cutoff 0.28 --n_iters 80 --lam 0.0001 --seeds 0 10 20 --save_probes --load_shared $SHARED"

until [ -f $SHARED/val_y.pt ] && [ -f $SHARED/W_c.pt ]; do sleep 300; done
sleep 600  # let the shared-stage process finish writing / move on to its seed-0 probe

jobs_list=()
for l in 1e-05 3e-05 0.0001 0.0003 0.001 0.003 0.01; do
  jobs_list+=("gated_$l|$COMMON -g --gate_lams $l --gate_forward hard --gate_epochs 150 --random_control")
done
for l in 1e-05 3e-05 0.0001 0.0003 0.001 0.003 0.01; do
  jobs_list+=("sparse_$l|${COMMON/--lam 0.0001/} --lams $l")
done

i=0
for job in "${jobs_list[@]}"; do
  name=${job%%|*}; args=${job#*|}
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do sleep 120; done
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  echo "$(date) start $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u train_cbm_seed_sweep.py $args > $LOG/lfcbm_imagenet_$name.log 2>&1 &
done
wait
echo "$(date) all LF-CBM ImageNet runs done"
