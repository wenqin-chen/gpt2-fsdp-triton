# gpt2-fsdp-triton

Pretraining GPT-2 small (124M parameters) on the FineWeb-Edu 10B-token sample with PyTorch
**DDP and FSDP2** on multiple GPUs, with honestly measured **tokens/s and MFU**, a held-out
comparison against OpenAI's GPT-2 124M on identical tokens, and a fused **Triton LayerNorm** kernel
benchmarked against PyTorch eager and `torch.compile`.

**Status: under construction.** Single-GPU, one-node (DDP, FSDP2) and two-node (FSDP2, 16 H200)
throughput runs and the Triton kernel benchmark are done; the full one-epoch pretraining run is
queued. All numbers live in [`RESULTS.md`](RESULTS.md), which `gptfsdp report` generates from the
run records in `results/runs/` (config, manifest, JSONL log per run) and `bench/*.json` - nothing
is typed by hand. The plan, recipe, metric definitions and honesty rules are in
[`SPEC.md`](SPEC.md). HellaSwag was dropped: its upstream repository is blocked by a DMCA notice
(see SPEC section 3).

## Layout

| Path | What |
|---|---|
| `src/gptfsdp/model.py` | GPT-2 (pre-LN, tied embeddings, SDPA attention), FLOP accounting |
| `src/gptfsdp/data.py` | `uint16` token shards and the distributed, resumable loader |
| `src/gptfsdp/dist.py` | torchrun process groups; DDP or FSDP2 (`fully_shard`) wrapping |
| `src/gptfsdp/train.py` | training loop, run directory, JSONL logs, throughput and MFU |
| `src/gptfsdp/checkpoint.py` | resume-safe sharded checkpoints (`torch.distributed.checkpoint`) |
| `src/gptfsdp/evaluate.py` | held-out loss for runs and for OpenAI's GPT-2 124M (same tokens) |
| `src/gptfsdp/hellaswag.py` | multiple-choice completion scoring (HellaSwag format; dataset dropped) |
| `src/gptfsdp/kernels/layernorm.py` | fused LayerNorm forward/backward in Triton |
| `src/gptfsdp/bench.py` | kernel benchmark (eager vs compile vs Triton) |
| `src/gptfsdp/prepare.py` | FineWeb-Edu download and tokenization into shards |
| `src/gptfsdp/report.py` | `RESULTS.md` from run logs |

## Quick start (CPU, tiny model)

```bash
uv sync --extra cpu --extra dev
uv run pytest -q
```

On a GPU node: `uv sync --extra cu130 --extra data`, then see `slurm/`.

## Licence

MIT. FineWeb-Edu is released under ODC-BY by Hugging Face; HellaSwag by its authors (MIT).
