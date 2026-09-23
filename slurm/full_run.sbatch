#!/bin/bash
# The full pretraining run: one epoch of the FineWeb-Edu 10B sample (18,722 steps x 524,288 tokens)
# on one node x 8 H200. Settings come from configs/full_run.env at start time; resubmitting the same
# RUN_NAME resumes from its newest checkpoint (time_limit_s checkpoints before the wall).
#   REPO=... DATA_DIR=... RUN_NAME=full_1x8 sbatch slurm/full_run.sbatch
#SBATCH --job-name=gpt2-full
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${REPO:?set REPO}"
# shellcheck disable=SC1091
source configs/full_run.env
export OMP_NUM_THREADS=2
echo "full run: parallel=$PARALLEL micro_batch=$MICRO_BATCH extra=$EXTRA git=$(git describe --always --dirty)"
# shellcheck disable=SC2086
srun .venv/bin/python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 \
  --master_addr 127.0.0.1 --master_port 29500 -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set parallel="$PARALLEL" micro_batch="$MICRO_BATCH" max_steps=18722 \
  run_name="${RUN_NAME:?set RUN_NAME}" data_dir="${DATA_DIR:?set DATA_DIR}" \
  time_limit_s=6600 resume=latest $EXTRA
