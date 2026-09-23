#!/bin/bash
# Throughput calibration on one node x 8 GPUs: short runs (60 steps, no validation, no final
# checkpoint), one configuration after another; a failing configuration does not stop the rest.
#   REPO=... DATA_DIR=... sbatch slurm/calibrate_1node.sbatch
#SBATCH --job-name=gpt2-cal
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:40:00
set -uo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
common=(max_steps=60 val_every=0 val_steps=0 ckpt_every=0 ckpt_final=false log_every=1
        warmup_timing_steps=10 compile=true data_dir="${DATA_DIR:?set DATA_DIR}")
run() {  # run <name> <key=value ...>
  local name=$1; shift
  echo "=== $name $(date -u +%H:%M:%S)"
  .venv/bin/python -m torch.distributed.run --nnodes 1 --nproc_per_node 8 \
    --master_addr 127.0.0.1 --master_port 29500 -m gptfsdp.cli train \
    --config configs/gpt2_124m.json --set "${common[@]}" run_name="$name" "$@" \
    || echo "=== $name FAILED"
}
run cal_ddp_mb64 parallel=ddp micro_batch=64
run cal_fsdp_mb64 parallel=fsdp micro_batch=64
run cal_fsdp_mb64_ac parallel=fsdp micro_batch=64 activation_checkpointing=true
run cal_ddp_mb32 parallel=ddp micro_batch=32
run cal_ddp_mb64_triton parallel=ddp micro_batch=64 norm=triton
