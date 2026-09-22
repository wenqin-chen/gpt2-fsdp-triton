# gpt2-fsdp-triton

Pretraining GPT-2 small (124M parameters) on the FineWeb-Edu 10B-token sample with PyTorch
**DDP and FSDP2** on multiple GPUs, with honestly measured **tokens/s and MFU**, a **HellaSwag**
evaluation, and a fused **Triton LayerNorm** kernel benchmarked against PyTorch eager and
`torch.compile`.

**Status: under construction.** No training run has been made yet; `RESULTS.md` is generated from
run logs by `gptfsdp report` and is the only place results will appear. The plan, recipe, metric
definitions and honesty rules are in [`SPEC.md`](SPEC.md).

## Layout

| Path | What |
|---|---|
| `src/gptfsdp/model.py` | GPT-2 (pre-LN, tied embeddings, SDPA attention), FLOP accounting |
| `src/gptfsdp/data.py` | `uint16` token shards and the distributed, resumable loader |
| `src/gptfsdp/dist.py` | torchrun process groups; DDP or FSDP2 (`fully_shard`) wrapping |
| `src/gptfsdp/train.py` | training loop, run directory, JSONL logs, throughput and MFU |
| `src/gptfsdp/checkpoint.py` | resume-safe sharded checkpoints (`torch.distributed.checkpoint`) |
| `src/gptfsdp/hellaswag.py` | HellaSwag validation scoring |
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
