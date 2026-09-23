#!/bin/bash
# One full H200 (not a MIG slice: slices distort timings): GPU tests, then the LayerNorm benchmark.
#SBATCH --job-name=gpt2-kernel
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${REPO:?set REPO}"
.venv/bin/python -m pytest -q -m gpu tests/test_kernel_gpu.py
.venv/bin/python -m gptfsdp.cli bench-layernorm --label "${LABEL:-}" \
  --out "bench/$(date +%Y%m%dT%H%M%S)_layernorm${LABEL:+_$LABEL}.json"
