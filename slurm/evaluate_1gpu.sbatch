#!/bin/bash
# Final evaluation on one GPU under the free debug QOS: held-out loss on the first 10,485,760 tokens
# of the validation shard (the slice the training loop validated on) and ARC-Easy (test), each for
# the run's newest checkpoint and for OpenAI's GPT-2 124M. Needs `gptfsdp download-evals` first.
#   REPO=... DATA_DIR=... EVALS=... RUN=runs/full_fsdp_2x8 sbatch slurm/evaluate_1gpu.sbatch
#SBATCH --job-name=gpt2-eval
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
set -euo pipefail
cd "${REPO:?set REPO}"
arc="${EVALS:?set EVALS}/ai2_arc/ARC-Easy/test-00000-of-00001.parquet"
py() { .venv/bin/python -m gptfsdp.cli "$@"; }
py heldout --run "${RUN:?set RUN}" --data "${DATA_DIR:?set DATA_DIR}" --tokens 10485760
py heldout --gpt2 "$EVALS/gpt2" --data "$DATA_DIR" --tokens 10485760 --out baselines/openai_gpt2_heldout.json
py arc-easy --run "$RUN" --data "$arc"
py arc-easy --gpt2 "$EVALS/gpt2" --data "$arc" --out baselines/openai_gpt2_arc_easy.json
