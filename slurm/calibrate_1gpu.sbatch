#!/bin/bash
# Single-GPU calibration under the free `debug` QOS (1 GPU, <= 1 h): the real-data pipeline and the
# per-GPU throughput that multi-GPU scaling is measured against. 30 steps per configuration.
#   REPO=... DATA_DIR=... sbatch slurm/calibrate_1gpu.sbatch
#SBATCH --job-name=gpt2-cal1
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
set -uo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=4
common=(max_steps=30 val_every=0 val_steps=0 ckpt_every=0 ckpt_final=false log_every=1
        warmup_timing_steps=10 compile=true data_dir="${DATA_DIR:?set DATA_DIR}")
run() {
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S)"
  .venv/bin/python -m gptfsdp.cli train --config configs/gpt2_124m.json \
    --set "${common[@]}" run_name="$name" "$@" || echo "=== $name FAILED"
}
run cal1_mb64 micro_batch=64
run cal1_mb64_triton micro_batch=64 norm=triton
run cal1_mb64_ac micro_batch=64 activation_checkpointing=true
run cal1_mb32 micro_batch=32
