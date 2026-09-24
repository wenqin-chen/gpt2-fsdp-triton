# gpt2-fsdp-triton — SPEC v0.1 (2026-09-22)

Pretrain a GPT-2-small-class (124M) language model on the FineWeb-Edu 10B-token sample with
PyTorch DDP and FSDP on multiple GPUs, report throughput and MFU honestly, evaluate held-out loss
against OpenAI's GPT-2 124M, and ship one custom Triton kernel benchmarked against PyTorch eager
and `torch.compile`.

This is project P2 of the gap-closing sprint: it exists to make the following claim true, with a
public repository as the receipt.

> Pretrained a 124M-parameter GPT-style LLM on the FineWeb-Edu 10B-token corpus with PyTorch FSDP
> across __ GPUs (bf16, activation checkpointing, resume-safe checkpoints), reaching __ tokens/s
> and __ % MFU, and wrote a fused Triton __ kernel that is __× faster than PyTorch eager.

Every blank is filled only from a run log (section 8).

## 1. Claimable milestone

1. Public repository, CI green (CPU tests).
2. A full 10B-token run (one epoch of the sample) with logged loss, tokens/s and MFU.
3. A two-node run (16 GPUs) with logged throughput, reported as scaling efficiency against one node.
4. Held-out loss of the final checkpoint and of OpenAI's GPT-2 124M on identical validation tokens
   (HellaSwag was planned; see section 3).
5. A kernel benchmark table (Triton vs eager vs `torch.compile`, forward and forward+backward,
   several shapes) plus the kernel's effect on end-to-end step time.

Status 2026-09-24: all five are done (README headline, RESULTS.md). Filling the claim above from
RESULTS.md changes two things. The full run did not use activation checkpointing; that was measured
on one GPU as a memory/throughput trade (`cal1_mb64` vs `cal1_mb64_ac`). The kernel beats eager or
`torch.compile` only at some shapes and slows the compiled model end to end, so the claim gives
the measured ranges instead of a single speed-up.

## 2. Non-goals

No new architecture research, no instruction tuning or RLHF, no model larger than GPT-2 medium,
no claims against public leaderboards, no "first"/"state-of-the-art" wording, no numbers typed by
hand into the README.

## 3. Data

- **Corpus:** `HuggingFaceFW/fineweb-edu`, subset `sample-10BT` (9.67M documents, ODC-BY licence).
  Tokenized with the GPT-2 BPE (`tiktoken`, `gpt2`), an `<|endoftext|>` token before every
  document, stored as `uint16` shards of 100M tokens; shard 0 is the validation shard and is never
  trained on. The download and tokenization run on the cluster (the development Mac has ~19 GB
  free); a manifest records the dataset revision, shard sizes and sha256.
- **HellaSwag — dropped (2026-09-23):** the upstream repository `rowanz/hellaswag` is blocked by a
  wikiHow DMCA notice of 2026-09-14 that lists it as an infringing dataset; the project does not
  fetch it from mirrors. The scoring code stays (tested on a fixture). Replacement, pending the
  owner's approval of the downloads: held-out loss on validation shard 0 for this model and for
  OpenAI's released GPT-2 124M (`openai-community/gpt2`, MIT) on identical tokens, optionally plus
  ARC-Easy (AI2, CC BY-SA 4.0) scored with the same completion harness for both models.
- **References (verified 2026-09-22):** llm.c's GPT-2 124M reproduction on plain FineWeb 10B
  tokens reached validation loss 3.29 and HellaSwag 29.9 % in ~90 min on 8×A100 80GB with up to
  ~60 % MFU (llm.c discussion #481). FineWeb and FineWeb-Edu validation losses are not
  comparable, so llm.c's loss is context only; the head-to-head comparison is this project's model
  against OpenAI's GPT-2 124M on the same FineWeb-Edu validation tokens.

## 4. Model and recipe

GPT-2 small: 12 layers, 12 heads, d_model 768, context 1024, vocabulary 50,257 padded to 50,304,
pre-LayerNorm, GELU (tanh), tied input/output embeddings, GPT-2 initialisation (residual
projections scaled by 1/sqrt(2·n_layer)), attention through `F.scaled_dot_product_attention`
(causal, FlashAttention backend on GPU).

| Item | Value |
|---|---|
| Tokens per optimizer step | 524,288 (2^19) = micro-batch × 1024 × world × grad-accumulation |
| Steps for one epoch | 18,722 (99 train shards; the loader drops each shard's tail < one step): 9,815,719,936 tokens |
| Optimizer | AdamW (fused on CUDA), β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 on ≥2-D tensors |
| Learning rate | 6e-4 peak, linear warm-up 715 steps, cosine to 6e-5 |
| Gradient clipping | 1.0 (global norm) |
| Precision | bf16 autocast (DDP) / bf16 parameters with fp32 reduction (FSDP) |
| Compilation | `torch.compile` on GPU (flag), eager on CPU |

## 5. Parallelism and scaling plan

1. **Single GPU:** correctness, the Triton kernel, profiler trace.
2. **DDP, 1 node × 8 GPUs:** baseline throughput and MFU.
3. **FSDP2 (`fully_shard`), 1 node × 8 GPUs:** same recipe; activation checkpointing on/off
   (memory vs step time); the full 10B-token run uses the faster of DDP / FSDP.
4. **FSDP2, 2 nodes × 8 GPUs:** `torchrun` under SLURM (`srun`, c10d rendezvous); scaling
   efficiency = tokens/s(16) / (2 × tokens/s(8)).

Checkpoints are resume-safe: model, optimizer, step, data-loader position and RNG state, written
with `torch.distributed.checkpoint` (sharded) every N steps; a resumed run must reproduce the
loss of an uninterrupted run at the next logged step (tested on CPU with two processes).

## 6. Metrics

- **tokens/s** = tokens processed per optimizer step / wall time of the step (CUDA-synchronised),
  median over logged steps after warm-up.
- **MFU** = (6·N + 12·L·H·Q·T) × tokens/s / (n_GPUs × peak bf16 dense FLOP/s), where N counts
  non-embedding + embedding parameters as in nanoGPT's `estimate_mfu`; the peak value and its
  source are logged with every run (e.g. H200 SXM and A100 per NVIDIA datasheets).
- Validation loss every 250 steps on a fixed slice of shard 0; the held-out comparison at the end.
- Memory: peak allocated per GPU.

## 7. Triton kernel

A fused LayerNorm (forward and backward, weight and bias gradients) as a
`torch.autograd.Function`, drop-in for `nn.LayerNorm` in the model. Correctness: max abs error
against `torch.nn.functional.layer_norm` in fp32 and bf16 within stated tolerances, gradient
check on small shapes. Benchmark: shapes (B·T, 768) and (B·T, 1024–4096), CUDA-event timing with
warm-up and repeats, reported as GB/s and speed-up over eager and `torch.compile`; end-to-end
step-time change when the model uses the kernel. Stretch goal: a fused linear + cross-entropy
kernel that never materialises the (B·T × vocab) logits.

## 8. Honesty rules

- Every published number comes from a run directory (`runs/<run_id>/`: config, git sha, package
  versions, hardware, JSONL log); `RESULTS.md` and the README tables are generated from those
  logs, never typed.
- MFU uses the formula above with the peak stated; throughput is measured after warm-up with
  synchronisation; any number from a partial run says so.
- Comparisons with other projects are stated with their source; the GPT-2 124M comparison is run
  by this project's code on the same tokens.

## 9. Compute (decided 2026-09-23: UW Tillicum)

Tillicum's H200 partition is oversubscribed (all healthy nodes allocated, dozens of multi-day
jobs pending): full-node jobs with walls of 40 min or more waited a day or longer, while 30-min
two-node jobs backfilled within hours. So the full run is 2 nodes x 8 H200 with FSDP2 in
resumable 30-min chunks (`slurm/full_run_2node.sbatch`), and each 8-GPU calibration configuration
is its own 10-min job. Single-GPU work uses the free `debug` QOS (1 GPU, 1 h).

The first chunk reached step 15,251 of 18,722; the second two-node chunk then waited days in the
queue. The remaining steps therefore go to `slurm/finish.sbatch`, queued in two shapes at once
(one node with a 20-min wall, two nodes with a 16-min wall); whichever starts first finishes the
run and evaluates it, and cancels the other. On one node the per-GPU micro-batch doubles (64
instead of 32), so every step still has 524,288 tokens, and DCP reshards the 16-rank checkpoint
onto 8 ranks. Each job logs a `chunk` record, and RESULTS.md gives one row per chunk, because
tokens/s depends on the GPU count. Only a run's first job writes `config.json`; later jobs
record what they changed.

Original options:

| Option | Hardware | Notes |
|---|---|---|
| UW Tillicum | 22 nodes × 8 H200; per job ≤ 16 GPUs, ≤ 24 h | already set up; H200 ≈ $0.90/GPU-hour, MIG slices ≈ $0.13 for development |
| NERSC Perlmutter | 4 × A100 per node | needs an allocation the user may use for this project |

Estimated Tillicum cost for the milestone: data preparation on a MIG slice (< $1), kernel and
single-GPU work (≈ $1), DDP/FSDP calibration runs on one node (≈ $4), the full 10B-token run on
8 H200 (≈ 1 h, ≈ $7), a two-node scaling run (≈ $4): about $15–20 with contingency.

## 10. Tests

CPU (CI): model shapes and parameter count (124.4M with tied embeddings), loss decreases on a tiny
config, data-loader determinism and resume, MFU arithmetic, learning-rate schedule, HellaSwag
scoring on a two-item fixture, 2-process gloo DDP and FSDP steps match a single-process step,
checkpoint save/resume reproduces the uninterrupted loss, a later chunk with half the micro-batch
(twice the accumulation) matches it too, and resuming a finished run changes nothing. GPU tests (`-m gpu`, run on the
cluster): Triton LayerNorm forward/backward against PyTorch, bf16 autocast step.
