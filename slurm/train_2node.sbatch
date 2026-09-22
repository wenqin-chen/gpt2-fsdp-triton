#!/bin/bash
# Two nodes x 8 GPUs with FSDP2 (Tillicum allows up to 16 GPUs per job). Usage as train_1node.
#SBATCH --job-name=gpt2-2x8
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun .venv/bin/python -m torch.distributed.run --nnodes 2 --nproc_per_node 8 \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$head_node:29500" \
  -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set parallel=fsdp run_name="${RUN_NAME:?set RUN_NAME}" data_dir="${DATA_DIR:?set DATA_DIR}" \
  time_limit_s="${TIME_LIMIT_S:-0}" "$@"
