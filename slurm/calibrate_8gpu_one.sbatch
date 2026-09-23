#!/bin/bash
# One 8-GPU calibration configuration per job (60 steps, no validation, no final checkpoint) with
# a 10-min wall, so each job can backfill on its own.
#   REPO=... DATA_DIR=... NAME=cal8_fsdp_mb64 SETTINGS="parallel=fsdp micro_batch=64" sbatch ...
#SBATCH --job-name=gpt2-cal8
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:10:00
set -euo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
# shellcheck disable=SC2086
srun .venv/bin/python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 \
  --master_addr 127.0.0.1 --master_port 29500 -m gptfsdp.cli train --config configs/gpt2_124m.json \
  --set max_steps=60 val_every=0 val_steps=0 ckpt_every=0 ckpt_final=false warmup_timing_steps=10 \
  compile=true run_name="${NAME:?set NAME}" data_dir="${DATA_DIR:?set DATA_DIR}" ${SETTINGS:?set SETTINGS}
