#!/bin/bash
# Weak-scaling measurement: 2 nodes x 8 H200 with FSDP2, the same per-GPU micro-batch (64) as the
# one-node runs, so tokens per step double (1,048,576); 100 steps, no validation, no checkpoint.
# Scaling efficiency = tokens/s(16) / (2 x tokens/s(8)) against the one-node FSDP calibration.
#   REPO=... DATA_DIR=... sbatch slurm/scaling_2node.sbatch
#SBATCH --job-name=gpt2-2x8
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
srun .venv/bin/python -m torch.distributed.run --nnodes 2 --nproc_per_node 8 \
  --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d --rdzv_endpoint "$head_node:29500" \
  -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set parallel=fsdp micro_batch=64 total_batch_tokens=1048576 max_steps=100 \
  val_every=0 val_steps=0 ckpt_every=0 ckpt_final=false warmup_timing_steps=10 \
  run_name=scaling_fsdp_2x8 data_dir="${DATA_DIR:?set DATA_DIR}"
