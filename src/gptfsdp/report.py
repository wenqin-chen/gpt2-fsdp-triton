"""RESULTS.md from run directories and benchmark files - no number is typed by hand.

Per run: parallel mode, world size, device, steps, tokens seen, median tokens/s and MFU over the
logged steps that are not flagged ``timing_warmup``, peak memory, final validation loss and the
latest HellaSwag record. Every row names its run id; the manifest's git sha and peak source are
listed under the table. LayerNorm benchmark files (``bench/*.json``) become a second table.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def read_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    records = read_log(run_dir)
    steps = [r for r in records if r.get("event") == "step"]
    timed = [r for r in steps if not r.get("timing_warmup")]
    vals = [r for r in records if r.get("event") == "val"]
    hs = [r for r in records if r.get("event") == "hellaswag"]
    mfus = [r["mfu"] for r in timed if r.get("mfu") is not None]
    mems = [r["max_mem_gb"] for r in steps if "max_mem_gb" in r]
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "status": manifest.get("status"),
        "parallel": manifest.get("parallel"),
        "world": manifest.get("world"),
        "device": manifest.get("device"),
        "steps": (steps[-1]["step"] + 1) if steps else 0,
        "tokens_seen": steps[-1]["tokens_seen"] if steps else 0,
        "tokens_per_s": statistics.median(r["tokens_per_s"] for r in timed) if timed else None,
        "mfu": statistics.median(mfus) if mfus else None,
        "n_timed_steps": len(timed),
        "max_mem_gb": max(mems) if mems else None,
        "final_val_loss": vals[-1]["val_loss"] if vals else None,
        "hellaswag_acc_norm": hs[-1]["acc_norm"] if hs else None,
        "hellaswag_n": hs[-1]["n"] if hs else None,
        "git_sha": manifest.get("git_sha"),
        "peak_source": manifest.get("peak_source"),
    }


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "—"
    return format(value, spec)


def runs_table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| run | status | parallel | GPUs | device | steps | tokens | tokens/s (median) | MFU | "
        "peak mem (GiB) | val loss |\n|" + "---|" * 11 + "\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| `{r['run_id']}` | {r['status']} | {r['parallel']} | {r['world']} | {r['device']} | "
            f"{r['steps']} | {r['tokens_seen']:,} | {_fmt(r['tokens_per_s'], ',.0f')} | "
            f"{_fmt(r['mfu'], '.1%')} | {_fmt(r['max_mem_gb'], '.1f')} | "
            f"{_fmt(r['final_val_loss'], '.4f')} |\n"
        )
    return head + body


def bench_table(bench: dict[str, Any]) -> str:
    head = (
        f"Device: {bench['device']} · torch {bench['torch']} · triton {bench['triton']}\n\n"
        "| M × N | dtype | eager fwd / fwd+bwd (ms) | compile fwd / fwd+bwd (ms) | "
        "Triton fwd / fwd+bwd (ms) | Triton fwd+bwd GB/s | speed-up vs eager (fwd+bwd) | "
        "speed-up vs compile (fwd+bwd) | max abs err (dx) |\n|" + "---|" * 9 + "\n"
    )
    body = ""
    for r in bench["rows"]:
        cells = [
            f"{r['M']} × {r['N']}", r["dtype"],
            *(
                f"{r[k]['fwd_ms']:.3f} / {r[k]['fwd_bwd_ms']:.3f}"
                for k in ("eager", "compile", "triton")
            ),
            f"{r['triton']['fwd_bwd_gbps']:,.0f}",
            f"{r['speedup_fwd_bwd_vs_eager']:.2f}×", f"{r['speedup_fwd_bwd_vs_compile']:.2f}×",
            f"{r['max_abs_err']['dx']:.2e}",
        ]  # fmt: skip
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


KEY_SHAPES: tuple[tuple[int, int], ...] = ((16384, 768), (65536, 768))

# Throughput scaling at a fixed per-GPU micro-batch of 64 x 1024 tokens (weak scaling): run ids of
# the single-GPU baseline and of the multi-GPU runs, in table order.
SCALING_BASELINE = "cal1_mb64"
SCALING_RUNS: tuple[str, ...] = ("cal8_ddp_mb64", "cal8_fsdp_mb64", "scaling_fsdp_2x8")


def scaling_table(summaries: dict[str, dict[str, Any]]) -> str:
    base = summaries.get(SCALING_BASELINE)
    if base is None or base.get("tokens_per_s") is None:
        return "_Needs the single-GPU baseline run._\n"
    head = (
        "| run | parallel | GPUs | tokens/s | speed-up vs 1 GPU | efficiency vs linear | MFU |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = [base] + [summaries[r] for r in SCALING_RUNS if r in summaries]
    body = ""
    for r in rows:
        speedup = r["tokens_per_s"] / base["tokens_per_s"]
        body += (
            f"| `{r['run_id']}` | {r['parallel']} | {r['world']} | {r['tokens_per_s']:,.0f} | "
            f"{speedup:.2f}× | {speedup / r['world']:.1%} | {_fmt(r['mfu'], '.1%')} |\n"
        )
    one, two = summaries.get("cal8_fsdp_mb64"), summaries.get("scaling_fsdp_2x8")
    if one and two and one.get("tokens_per_s") and two.get("tokens_per_s"):
        eff = two["tokens_per_s"] / (2 * one["tokens_per_s"])
        body += (
            f"\nOne node to two nodes with FSDP2 (8 → 16 GPUs, same per-GPU micro-batch): "
            f"{eff:.1%} weak-scaling efficiency.\n"
        )
    return head + body


def iterations_table(benches: list[tuple[str, dict[str, Any]]]) -> str:
    """One row per benchmark file: forward+backward ms at the GPT-2 training shapes."""
    shape_cols = " | ".join(f"{m} × {n}: Triton / eager / compile (ms)" for m, n in KEY_SHAPES)
    head = f"| file | variant | git sha | {shape_cols} |\n|" + "---|" * (3 + len(KEY_SHAPES)) + "\n"
    body = ""
    for name, bench in benches:
        by_shape = {(r["M"], r["N"]): r for r in bench["rows"]}
        cells = []
        for shape in KEY_SHAPES:
            r = by_shape.get(shape)
            if r is None:
                cells.append("—")
            else:
                cells.append(
                    " / ".join(f"{r[k]['fwd_bwd_ms']:.3f}" for k in ("triton", "eager", "compile"))
                )
        raw = str(bench.get("git_sha") or "—")
        sha = raw[:7] + (" (dirty)" if raw.endswith("dirty") else "")
        body += f"| `{name}` | {bench.get('label') or '—'} | {sha} | " + " | ".join(cells) + " |\n"
    return head + body


def write_results(
    runs: str | Path, out: str | Path, bench_dir: str | Path = "bench"
) -> dict[str, Any]:
    runs_dir = Path(runs)
    rows = [
        s
        for d in sorted(p for p in runs_dir.iterdir() if p.is_dir())
        if (s := summarize_run(d)) is not None
    ] if runs_dir.is_dir() else []  # fmt: skip
    src = runs_dir.as_posix()
    parts = [
        f"# Results\n\n_Generated by `gptfsdp report` from `{src}/*/manifest.json`, "
        f"`{src}/*/log.jsonl` and `bench/*.json`; do not edit by hand._\n\n"
        "## Pretraining runs\n\n"
    ]
    parts.append(runs_table(rows) if rows else "_No runs yet._\n")
    parts.append("\n## Throughput scaling (micro-batch 64 per GPU, bf16, torch.compile)\n\n")
    parts.append(scaling_table({r["run_id"]: r for r in rows}))
    if rows:
        sources = sorted({r["peak_source"] for r in rows if r["peak_source"]})
        parts.append("\nMFU peaks: " + "; ".join(sources) + ".\n" if sources else "")
    parts.append("\n## Held-out loss (validation shard 0, identical tokens for every model)\n\n")
    heldout = [
        (f"`{r['run_id']}`", rec)
        for r in rows
        for rec in read_log(runs_dir / r["run_id"])
        if rec.get("event") == "heldout"
    ]
    baseline = Path("baselines/openai_gpt2_heldout.json")
    if baseline.is_file():
        rec = json.loads(baseline.read_text())
        heldout.append((rec["model"], rec))
    if heldout:
        parts.append("| model | checkpoint | tokens | loss | perplexity |\n|---|---|---|---|---|\n")
        for name, rec in heldout:
            parts.append(
                f"| {name} | {rec.get('checkpoint', '—')} | {rec['tokens']:,} | "
                f"{rec['val_loss']:.4f} | {rec['perplexity']:.2f} |\n"
            )
    else:
        parts.append("_Not run yet._\n")
    files = sorted(Path(bench_dir).glob("*.json")) if Path(bench_dir).is_dir() else []
    benches = [(p.name, json.loads(p.read_text())) for p in files]
    parts.append("\n## Triton LayerNorm benchmark\n\n")
    if benches:
        name, latest = benches[-1]
        parts.append(f"Latest: `{name}` ({latest.get('label') or 'unlabelled'}).\n\n")
        parts.append(bench_table(latest))
        parts.append("\n### Kernel iterations (forward + backward)\n\n" + iterations_table(benches))
        by_id = {r["run_id"]: r for r in rows}
        plain, fused = by_id.get("cal1_mb64"), by_id.get("cal1_mb64_triton")
        if plain and fused and plain.get("tokens_per_s") and fused.get("tokens_per_s"):
            change = fused["tokens_per_s"] / plain["tokens_per_s"] - 1
            tps_fused, tps_plain = fused["tokens_per_s"], plain["tokens_per_s"]
            parts.append(
                "\nEnd to end (one H200, torch.compile, micro-batch 64): the model with the "
                f"Triton LayerNorm runs at {tps_fused:,.0f} tokens/s against {tps_plain:,.0f} "
                f"with `nn.LayerNorm` ({change:+.1%}); `torch.compile` fuses LayerNorm into its "
                "own kernels, and the custom autograd function breaks that fusion.\n"
            )
    else:
        parts.append("_Not run yet._\n")
    path = Path(out)
    path.write_text("".join(parts))
    return {"out": str(path), "n_runs": len(rows), "n_benchmarks": len(benches)}
