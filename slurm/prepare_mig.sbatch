#!/bin/bash
# Tokenize FineWeb-Edu sample-10BT into uint16 shards on a MIG slice (2 CPUs). The Hub download
# inside `prepare` needs outbound internet; if compute nodes lack it, run the download on the login
# node first (same command, it resumes and skips finished files).
#SBATCH --job-name=gpt2-prep
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${REPO:?set REPO}"
.venv/bin/python -m gptfsdp.cli prepare --out "${DATA_DIR:?set DATA_DIR}" --workers 2
