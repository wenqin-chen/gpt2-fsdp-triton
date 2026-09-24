from __future__ import annotations

import json
from pathlib import Path

import torch

from gptfsdp.hellaswag import evaluate, predict, render
from gptfsdp.report import summarize_run, write_results

VOCAB = 16


def encode(text: str) -> list[int]:
    """Toy tokenizer: one token per word, id = word length (1..15)."""
    return [min(len(w), VOCAB - 1) for w in text.split()]


class Oracle(torch.nn.Module):
    """Predicts the next token as 'same as the current token' with high confidence."""

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):  # type: ignore[no-untyped-def]
        logits = torch.full((*idx.shape, VOCAB), -10.0)
        logits.scatter_(-1, idx.unsqueeze(-1), 10.0)
        return logits, None


ITEMS = [
    # context words have length 3; the right ending repeats length-3 words
    {
        "ctx": "the cat sat",
        "endings": ["ran far away", "hop hop hop", "extraordinary", "a"],
        "label": 1,
    },
    {"ctx": "abcd abcd", "endings": ["xyz", "q", "wxyz wxyz", "hello world"], "label": 2},
]


def test_render_masks_only_the_ending() -> None:
    tokens, mask, label = render(ITEMS[0], encode)
    assert tokens.shape[0] == 4 and label == 1
    assert mask[1].tolist()[:3] == [0, 0, 0] and mask[1].sum().item() == 3
    assert mask[2].sum().item() == 1 and tokens[2, 3].item() == min(len("extraordinary"), VOCAB - 1)


def test_predict_and_evaluate_with_an_oracle_model() -> None:
    tokens, mask, _ = render(ITEMS[0], encode)
    logits, _ = Oracle()(tokens)
    assert predict(logits, tokens, mask)[1] == 1
    result = evaluate(Oracle(), ITEMS, encode, torch.device("cpu"))
    assert result == {"n": 2, "acc": 1.0, "acc_norm": 1.0}
    sharded = evaluate(Oracle(), ITEMS, encode, torch.device("cpu"), rank=0, world=2)
    assert sharded["n"] == 1  # without a process group each rank only reports its own share


def _fake_run(root: Path, name: str, world: int, tps: list[float]) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": name, "status": "ok", "parallel": "fsdp", "world": world, "device": "NVIDIA H200",
        "git_sha": "abc", "peak_source": "NVIDIA H200 SXM datasheet",
    }))  # fmt: skip
    lines = [
        {"event": "step", "step": i, "loss": 5.0 - i, "tokens_per_s": t, "mfu": t / 1e7,
         "tokens_seen": (i + 1) * 1000, "timing_warmup": i == 0, "max_mem_gb": 10.0 + i}
        for i, t in enumerate(tps)
    ]  # fmt: skip
    lines.append({"event": "val", "step": len(tps), "val_loss": 3.25})
    lines.append({"event": "hellaswag", "acc_norm": 0.301, "acc": 0.29, "n": 10042, "step": 3})
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    return run


def test_report_uses_only_logged_numbers(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _fake_run(runs, "a_8gpu", 8, [1.0, 2e6, 3e6, 4e6])  # the warm-up step (1.0) is excluded
    s = summarize_run(runs / "a_8gpu")
    assert (
        s is not None and s["tokens_per_s"] == 3e6 and s["mfu"] == 0.3 and s["n_timed_steps"] == 3
    )
    assert s["max_mem_gb"] == 13.0 and s["final_val_loss"] == 3.25 and len(s["chunks"]) == 1
    assert s["chunks"][0]["world"] == 8  # a log without chunk records: the manifest describes it
    bench = tmp_path / "bench"
    bench.mkdir()
    timing = {"fwd_ms": 0.1, "fwd_bwd_ms": 0.3, "fwd_bwd_gbps": 900.0}
    row = {"M": 16384, "N": 768, "dtype": "bfloat16", "max_abs_err": {"dx": 1e-3},
           "speedup_fwd_bwd_vs_eager": 2.5, "speedup_fwd_bwd_vs_compile": 1.1,
           **{k: timing for k in ("eager", "compile", "triton")}}  # fmt: skip
    (bench / "h200.json").write_text(
        json.dumps({"device": "NVIDIA H200", "torch": "2.14", "triton": "3.8", "rows": [row]})
    )
    out = write_results(runs, tmp_path / "RESULTS.md", bench_dir=bench)
    text = (tmp_path / "RESULTS.md").read_text()
    assert out["n_runs"] == 1 and out["n_benchmarks"] == 1
    assert (
        "`a_8gpu`" in text
        and "3,000,000" in text
        and "30.0%" in text
        and "3.2500" in text  # final validation loss (the HellaSwag column was dropped)
    )
    assert "16384 × 768" in text and "2.50×" in text and "do not edit by hand" in text


def test_a_run_trained_by_several_jobs_gets_one_row_per_job(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = _fake_run(runs, "full", 8, [1.0, 2e6, 2e6, 2e6])  # job 1 wrote no chunk record
    manifest = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps({**manifest, "tokens_per_step": 1000}))
    more = [
        {"event": "chunk", "retroactive": True, "start_step": 0, "world": 16, "parallel": "fsdp",
         "device": "NVIDIA H200", "micro_batch": 32, "slurm_job_id": "101"},
        {"event": "resume", "step": 4},
        {"event": "chunk", "start_step": 4, "world": 8, "parallel": "fsdp",
         "device": "NVIDIA H200", "micro_batch": 64, "slurm_job_id": "202"},
        *({"event": "step", "step": i, "tokens_per_s": 1e6, "mfu": 0.25, "timing_warmup": i == 4,
           "tokens_seen": (i + 1) * 1000, "max_mem_gb": 20.0} for i in range(4, 7)),
        {"event": "val", "step": 7, "val_loss": 3.0},
    ]  # fmt: skip
    with (run / "log.jsonl").open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in more))
    s = summarize_run(run)
    assert s is not None and s["steps"] == 7 and s["final_val_loss"] == 3.0
    one, two = s["chunks"]
    assert (one["world"], one["steps"], one["tokens_per_s"]) == (16, 4, 2e6)
    assert (two["world"], two["first_step"], two["tokens_per_s"], two["mfu"]) == (8, 4, 1e6, 0.25)
    write_results(runs, tmp_path / "RESULTS.md", bench_dir=tmp_path / "none")
    text = (tmp_path / "RESULTS.md").read_text()
    assert "`full` chunk 1/2 | partial | fsdp | 16 |" in text
    assert "`full` chunk 2/2 | ok | fsdp | 8 |" in text and "1,000,000" in text
    assert "SLURM job 101: 16 GPUs, micro-batch 32, steps 0–3" in text
    assert "SLURM job 202: 8 GPUs, micro-batch 64, steps 4–6" in text and "1,000 tokens" in text
