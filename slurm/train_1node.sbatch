#!/bin/bash
# One node, 8 GPUs: DDP or FSDP2 (PARALLEL=ddp|fsdp). Usage:
#   REPO=/path/to/gpt2-fsdp-triton DATA_DIR=/path/to/shards RUN_NAME=fsdp_1x8 PARALLEL=fsdp \
#     sbatch --time=01:30:00 slurm/train_1node.sbatch [--set key=value ...]
#SBATCH --job-name=gpt2-1x8
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
srun .venv/bin/python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 \
  --master_addr 127.0.0.1 --master_port 29500 \
  -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set parallel="${PARALLEL:-fsdp}" run_name="${RUN_NAME:?set RUN_NAME}" data_dir="${DATA_DIR:?set DATA_DIR}" \
  time_limit_s="${TIME_LIMIT_S:-0}" "$@"
