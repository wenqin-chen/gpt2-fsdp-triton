"""RESULTS.md from run directories and benchmark files - no number is typed by hand.

Per run: parallel mode, world size, device, steps, tokens seen, median tokens/s and MFU over the
logged steps that are not flagged ``timing_warmup``, peak memory and final validation loss. A run
that several jobs trained ("chunks", see ``train.py``) gets one row per chunk, because tokens/s
depends on the number of GPUs. Every row names its run id; the manifest's git sha and peak source
are listed under the table. LayerNorm benchmark files (``bench/*.json``) become their own tables.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

CHUNK_FIELDS = ("world", "parallel", "device", "micro_batch", "grad_accum", "slurm_job_id")


def read_log(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "log.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_chunks(
    records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """A run's log as (chunk record, records the chunk wrote) pairs, in log order.

    A ``chunk`` record opens a chunk. Records before the first one were written by a job that
    predates chunk records; the ``retroactive`` chunk record a later job rebuilt describes them.
    A log without chunk records is one chunk with an empty description.
    """
    retro = next((r for r in records if r.get("event") == "chunk" and r.get("retroactive")), {})
    chunks: list[tuple[dict[str, Any], list[dict[str, Any]]]] = [(retro, [])]
    for r in records:
        if r.get("event") != "chunk":
            chunks[-1][1].append(r)
        elif not r.get("retroactive"):
            if chunks[-1][0] or chunks[-1][1]:
                chunks.append((r, []))
            else:
                chunks[-1] = (r, [])
    return chunks


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [r for r in records if r.get("event") == "step"]
    timed = [r for r in steps if not r.get("timing_warmup")]
    vals = [r for r in records if r.get("event") == "val"]
    mfus = [r["mfu"] for r in timed if r.get("mfu") is not None]
    mems = [r["max_mem_gb"] for r in steps if "max_mem_gb" in r]
    return {
        "first_step": steps[0]["step"] if steps else None,
        "steps": (steps[-1]["step"] + 1) if steps else 0,
        "tokens_seen": steps[-1]["tokens_seen"] if steps else 0,
        "tokens_per_s": statistics.median(r["tokens_per_s"] for r in timed) if timed else None,
        "mfu": statistics.median(mfus) if mfus else None,
        "n_timed_steps": len(timed),
        "max_mem_gb": max(mems) if mems else None,
        "final_val_loss": vals[-1]["val_loss"] if vals else None,
    }


def summarize_run(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    records = read_log(run_dir)
    parts = [(meta, recs) for meta, recs in split_chunks(records) if _stats(recs)["steps"]]
    chunks = []
    for meta, recs in parts:
        # a lone chunk without a record is described by the manifest; in a multi-chunk run the
        # manifest describes only the last chunk, so an undescribed chunk stays undescribed
        source = meta or (manifest if len(parts) == 1 else {})
        chunks.append({**{k: source.get(k) for k in CHUNK_FIELDS}, **_stats(recs)})
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "status": manifest.get("status"),
        "parallel": manifest.get("parallel"),
        "world": manifest.get("world"),
        "device": manifest.get("device"),
        **_stats(records),
        "chunks": chunks,
        "tokens_per_step": manifest.get("tokens_per_step"),
        "git_sha": manifest.get("git_sha"),
        "peak_source": manifest.get("peak_source"),
    }


def _with_se(p: float, n: int) -> str:
    """An accuracy with its binomial standard error."""
    return f"{p:.4f} ± {math.sqrt(p * (1 - p) / n):.4f}"


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "—"
    return format(value, spec)


def runs_table(runs: list[dict[str, Any]]) -> str:
    head = (
        "| run | status | parallel | GPUs | device | steps | tokens | tokens/s (median) | MFU | "
        "peak mem (GiB) | val loss |\n|" + "---|" * 11 + "\n"
    )
    body = ""
    for run in runs:
        chunks = run["chunks"] if len(run["chunks"]) > 1 else [run]
        for i, r in enumerate(chunks, start=1):
            name, status = f"`{run['run_id']}`", run["status"]
            if len(chunks) > 1:
                name += f" chunk {i}/{len(chunks)}"
                status = status if i == len(chunks) else "partial"
            body += (
                f"| {name} | {status} | {r['parallel']} | {r['world']} | {r['device']} | "
                f"{r['steps']:,} | {r['tokens_seen']:,} | {_fmt(r['tokens_per_s'], ',.0f')} | "
                f"{_fmt(r['mfu'], '.1%')} | {_fmt(r['max_mem_gb'], '.1f')} | "
                f"{_fmt(r['final_val_loss'], '.4f')} |\n"
            )
    return head + body


def chunk_notes(runs: list[dict[str, Any]]) -> str:
    """One sentence per multi-chunk run: which job trained which steps on how many GPUs."""
    notes = ""
    for run in runs:
        chunks = run["chunks"]
        if len(chunks) < 2:
            continue
        jobs = "; ".join(
            f"SLURM job {c['slurm_job_id'] or '—'}: {c['world']} GPUs, micro-batch "
            f"{c['micro_batch'] or '—'}, steps {c['first_step']:,}–{c['steps'] - 1:,}"
            for c in chunks
        )
        notes += (
            f"\n`{run['run_id']}` was trained by {len(chunks)} jobs that resumed one another from "
            f"sharded checkpoints ({jobs}); every step used "
            f"{_fmt(run['tokens_per_step'], ',')} tokens, and the checkpoint was resharded "
            "whenever the number of GPUs changed.\n"
        )
    return notes


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


# The one-epoch pretraining run that the README headline describes.
FULL_RUN = "full_fsdp_2x8"
HEADLINE_START, HEADLINE_END = "<!-- headline:start", "<!-- headline:end -->"


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _last(records: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    found = [r for r in records if r.get("event") == event]
    return found[-1] if found else None


def headline(
    rows: list[dict[str, Any]], runs_dir: Path, baselines: Path,
    benches: list[tuple[str, dict[str, Any]]],
) -> str:  # fmt: skip
    """The README summary: one bullet per result, from the same files as RESULTS.md."""
    by_id = {r["run_id"]: r for r in rows}
    bullets = []
    full = by_id.get(FULL_RUN)
    if full:
        chunks = "; ".join(
            f"{c['world']} GPUs for steps {c['first_step']:,}–{c['steps'] - 1:,} at "
            f"{_fmt(c['tokens_per_s'], ',.0f')} tokens/s ({_fmt(c['mfu'], '.1%')} MFU)"
            for c in full["chunks"]
        )
        done = "one epoch" if full["status"] == "ok" else f"status: {full['status']}"
        bullets.append(
            f"**Pretraining** ({done}): GPT-2 124M on {full['tokens_seen']:,} FineWeb-Edu tokens "
            f"({full['steps']:,} steps of {_fmt(full['tokens_per_step'], ',')}) with FSDP2 on "
            f"H200s: {chunks}. Final validation loss {_fmt(full['final_val_loss'], '.4f')}."
        )
        records = read_log(runs_dir / FULL_RUN)
        held, arc = _last(records, "heldout"), _last(records, "arc_easy")
        gpt2_held = _read_json(baselines / "openai_gpt2_heldout.json")
        gpt2_arc = _read_json(baselines / "openai_gpt2_arc_easy.json")
        if held and gpt2_held:
            bullets.append(
                f"**Held-out loss** on the same {held['tokens']:,} FineWeb-Edu validation tokens: "
                f"{held['val_loss']:.4f} against {gpt2_held['val_loss']:.4f} for OpenAI's GPT-2 "
                f"124M ({held['checkpoint']}; the run's training distribution, not GPT-2's)."
            )
        if arc and gpt2_arc:
            bullets.append(
                f"**ARC-Easy** ({arc['n']:,} test questions, zero-shot, one harness for both): "
                f"acc_norm {arc['acc_norm']:.3f} against {gpt2_arc['acc_norm']:.3f} for GPT-2 "
                f"124M, acc {arc['acc']:.3f} against {gpt2_arc['acc']:.3f}."
            )
    base, ddp = by_id.get(SCALING_BASELINE), by_id.get("cal8_ddp_mb64")
    one, two = by_id.get("cal8_fsdp_mb64"), by_id.get("scaling_fsdp_2x8")
    if all(r and r.get("tokens_per_s") for r in (base, ddp, one, two)):
        assert base and ddp and one and two
        ddp_eff = ddp["tokens_per_s"] / (ddp["world"] * base["tokens_per_s"])
        weak = two["tokens_per_s"] / (2 * one["tokens_per_s"])
        bullets.append(
            f"**Scaling**: DDP on {ddp['world']} H200 at {ddp_eff:.1%} of linear "
            f"({ddp['tokens_per_s']:,.0f} tokens/s); FSDP2 from one node to two at {weak:.1%} "
            f"weak-scaling efficiency ({two['tokens_per_s']:,.0f} tokens/s on {two['world']} GPUs)."
        )
    plain, fused = by_id.get("cal1_mb64"), by_id.get("cal1_mb64_triton")
    if benches and plain and fused and plain.get("tokens_per_s") and fused.get("tokens_per_s"):
        kernel_rows = benches[-1][1]["rows"]
        eager = [r["speedup_fwd_bwd_vs_eager"] for r in kernel_rows]
        comp = [r["speedup_fwd_bwd_vs_compile"] for r in kernel_rows]
        change = fused["tokens_per_s"] / plain["tokens_per_s"] - 1
        bullets.append(
            f"**Triton LayerNorm** (fused forward and backward, {len(kernel_rows)} shapes): "
            f"{min(eager):.2f}–{max(eager):.2f}× PyTorch eager and "
            f"{min(comp):.2f}–{max(comp):.2f}× `torch.compile` in isolation; inside the compiled "
            "model it changes tokens/s by "
            f"{change:+.1%} (the custom op breaks compile's own fusion), so training uses "
            "`nn.LayerNorm`."
        )
    return "".join(f"- {b}\n" for b in bullets)


def write_headline(readme: Path, text: str) -> bool:
    """Replace the README's marked headline block; False if the markers are missing."""
    doc = readme.read_text() if readme.is_file() else ""
    start, end = doc.find(HEADLINE_START), doc.find(HEADLINE_END)
    if start < 0 or end < start:
        return False
    body_start = doc.index("\n", start) + 1
    readme.write_text(doc[:body_start] + text + doc[end:])
    return True


def write_results(
    runs: str | Path, out: str | Path, bench_dir: str | Path = "bench",
    baselines: str | Path = "baselines", readme: str | Path | None = None,
) -> dict[str, Any]:  # fmt: skip
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
    parts.append(runs_table(rows) + chunk_notes(rows) if rows else "_No runs yet._\n")
    parts.append("\n## Throughput scaling (micro-batch 64 per GPU, bf16, torch.compile)\n\n")
    parts.append(scaling_table({r["run_id"]: r for r in rows}))
    if rows:
        sources = sorted({r["peak_source"] for r in rows if r["peak_source"]})
        parts.append("\nMFU peaks: " + "; ".join(sources) + ".\n" if sources else "")
    parts.append(
        "\n## Held-out loss (validation shard 0, identical tokens for every model)\n\n"
        "The validation shard is FineWeb-Edu, the distribution the runs here were trained on; "
        "OpenAI's GPT-2 was trained on WebText. A lower loss here shows the run fits its own "
        "distribution, not that it is the better model in general; ARC-Easy below is the check "
        "outside the training distribution.\n\n"
    )
    heldout = [
        (f"`{r['run_id']}`", rec)
        for r in rows
        for rec in read_log(runs_dir / r["run_id"])
        if rec.get("event") == "heldout"
    ]
    baseline = Path(baselines) / "openai_gpt2_heldout.json"
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
    parts.append(
        "\n## ARC-Easy (test, zero-shot, same harness for every model)\n\n"
        "Prompt `Question: …\\nAnswer:` + ` choice`; `acc_norm` picks the lowest mean token loss, "
        "`acc` the lowest summed loss; ± is the binomial standard error sqrt(p(1-p)/n). Compare "
        "rows with each other, not with other harnesses.\n\n"
    )
    arc = [
        (f"`{r['run_id']}`", rec)
        for r in rows
        for rec in read_log(runs_dir / r["run_id"])
        if rec.get("event") == "arc_easy"
    ]
    arc_baseline = Path(baselines) / "openai_gpt2_arc_easy.json"
    if arc_baseline.is_file():
        rec = json.loads(arc_baseline.read_text())
        arc.append((rec["model"], rec))
    if arc:
        parts.append("| model | checkpoint | questions | acc_norm | acc |\n|---|---|---|---|---|\n")
        for name, rec in arc:
            parts.append(
                f"| {name} | {rec.get('checkpoint', '—')} | {rec['n']:,} | "
                f"{_with_se(rec['acc_norm'], rec['n'])} | {_with_se(rec['acc'], rec['n'])} |\n"
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
    summary: dict[str, Any] = {"out": str(path), "n_runs": len(rows), "n_benchmarks": len(benches)}
    if readme is not None:
        text = headline(rows, runs_dir, Path(baselines), benches)
        summary["readme_headline"] = write_headline(Path(readme), text)
    return summary
