#!/bin/bash
# Finish a chunked FSDP2 run on whatever this job was given (one node or two, 8 H200 each), then
# evaluate it: held-out loss and ARC-Easy for the final checkpoint and, if still missing, for
# OpenAI's GPT-2 124M. Copies with different shapes can wait in the queue side by side; the first
# to start does the work. A copy that starts while another one runs exits at once, and the copy
# that completes the evaluation cancels the copies still pending.
# The global batch stays 524,288 tokens: micro-batch 524288 / (1024 x GPUs), at most 64 per GPU.
#   REPO=... DATA_DIR=... EVALS=... RUN_NAME=full_fsdp_2x8 TIME_LIMIT_S=840 \
#     sbatch --nodes=1 --time=00:22:00 slurm/finish.sbatch
#SBATCH --job-name=gpt2-finish
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:22:00
set -euo pipefail
cd "${REPO:?set REPO}"
export OMP_NUM_THREADS=2
run="runs/${RUN_NAME:?set RUN_NAME}"
max_steps=${MAX_STEPS:-18722}
final="$run/ckpt/step_$(printf %07d "$max_steps")/extra.json"
arc="${EVALS:?set EVALS}/ai2_arc/ARC-Easy/test-00000-of-00001.parquet"
py() { .venv/bin/python -m gptfsdp.cli "$@"; }
logged() { [ -f "$run/log.jsonl" ] && grep "\"event\": \"$1\"" "$run/log.jsonl" | grep -q "\"step\": $max_steps"; }
evaluated() {
  logged heldout && logged arc_easy &&
    [ -f baselines/openai_gpt2_heldout.json ] && [ -f baselines/openai_gpt2_arc_easy.json ]
}
# (queue lookups fail open: without squeue this job simply does the work)
others() { squeue --me --name="$SLURM_JOB_NAME" -h -o "%i %T %S" 2>/dev/null | awk -v me="$SLURM_JOB_ID" '$1 != me'; }

# another copy that started earlier (or in the same second with a lower job id) is the one working
mine=$(squeue -j "$SLURM_JOB_ID" -h -o %S 2>/dev/null || true)
while read -r id state start; do
  case $state in RUNNING | CONFIGURING) ;; *) continue ;; esac
  [ -n "$mine" ] || continue
  if [[ $start < $mine || ($start == "$mine" && $id -lt $SLURM_JOB_ID) ]]; then
    echo "job $id is already finishing $RUN_NAME; nothing to do"
    exit 0
  fi
done < <(others)
echo "finishing $RUN_NAME on $SLURM_JOB_NODELIST git=$(git describe --always --dirty)"

if [ ! -f "$final" ]; then
  per_node=$((${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}))
  micro=$((524288 / 1024 / (SLURM_NNODES * per_node)))
  ((micro > 64)) && micro=64
  head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  srun .venv/bin/python -m torch.distributed.run --nnodes "$SLURM_NNODES" \
    --nproc_per_node "$per_node" --rdzv_id "$SLURM_JOB_ID" --rdzv_backend c10d \
    --rdzv_endpoint "$head_node:29500" -m gptfsdp.cli train --config configs/gpt2_124m.json \
    --set parallel=fsdp micro_batch="$micro" max_steps="$max_steps" run_name="$RUN_NAME" \
    data_dir="${DATA_DIR:?set DATA_DIR}" time_limit_s="${TIME_LIMIT_S:-840}" resume=latest
fi

if [ -f "$final" ] && ! evaluated; then
  # two GPUs of this node: the run's evaluations (they append to its log) on one, GPT-2's on the other
  devices=${CUDA_VISIBLE_DEVICES:-0,1}
  read -r gpu_a gpu_b _ <<<"${devices//,/ }"
  (
    export CUDA_VISIBLE_DEVICES=$gpu_a
    logged heldout || py heldout --run "$run" --data "$DATA_DIR" --tokens 10485760
    logged arc_easy || py arc-easy --run "$run" --data "$arc"
  ) &
  run_evals=$!
  (
    export CUDA_VISIBLE_DEVICES=$gpu_b
    [ -f baselines/openai_gpt2_heldout.json ] || py heldout --gpt2 "$EVALS/gpt2" \
      --data "$DATA_DIR" --tokens 10485760 --out baselines/openai_gpt2_heldout.json
    [ -f baselines/openai_gpt2_arc_easy.json ] || py arc-easy --gpt2 "$EVALS/gpt2" \
      --data "$arc" --out baselines/openai_gpt2_arc_easy.json
  ) &
  gpt2_evals=$!
  wait "$run_evals"
  wait "$gpt2_evals"
fi

if evaluated; then
  others | awk '$2 == "PENDING" {print $1}' | xargs -r scancel || true
  echo "$RUN_NAME is complete and evaluated"
fi
