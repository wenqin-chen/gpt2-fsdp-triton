#!/bin/bash
# The full pretraining run on 2 nodes x 8 H200 with FSDP2, in resumable chunks: each job trains
# for time_limit_s (25 min), checkpoints and exits inside a 30-min wall (short walls backfill on a
# busy partition); the next job with the same RUN_NAME resumes from the newest checkpoint.
# Recipe: 524,288 tokens per step (micro-batch 32 x 1024 x 16 GPUs), 18,722 steps = one epoch.
#   REPO=... DATA_DIR=... RUN_NAME=full_fsdp_2x8 sbatch slurm/full_run_2node.sbatch
#SBATCH --job-name=gpt2-full2
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "chunk of ${RUN_NAME:?set RUN_NAME} on $SLURM_JOB_NODELIST git=$(git describe --always --dirty)"
srun .venv/bin/python -m torch.distributed.run --nnodes 2 --nproc_per_node 8 \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$head_node:29500" \
  -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set parallel=fsdp micro_batch=32 max_steps=18722 run_name="$RUN_NAME" \
  data_dir="${DATA_DIR:?set DATA_DIR}" time_limit_s=1500 resume=latest
