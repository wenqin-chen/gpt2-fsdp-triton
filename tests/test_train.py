from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from gptfsdp.data import synthetic_corpus
from gptfsdp.train import TrainConfig, train

REPO = Path(__file__).resolve().parents[1]


def tiny_cfg(tmp_path: Path, **kw: object) -> TrainConfig:
    data = tmp_path / "data"
    if not data.exists():
        synthetic_corpus(data, n_train_shards=2, shard_tokens=8192, vocab=256)
    base: dict[str, object] = dict(
        data_dir=str(data), out_dir=str(tmp_path / "runs"), preset="tiny", vocab_size=256,
        seq_len=32, micro_batch=4, total_batch_tokens=256, max_steps=30, warmup_steps=5,
        max_lr=3e-3, device="cpu", bf16=False, val_every=10, val_steps=2, ckpt_every=0,
        warmup_timing_steps=2,
    )  # fmt: skip
    base.update(kw)
    return TrainConfig(**base)  # type: ignore[arg-type]


def read_log(run_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]


def losses(run_dir: Path) -> dict[int, float]:
    return {int(r["step"]): float(r["loss"]) for r in read_log(run_dir) if r["event"] == "step"}  # type: ignore[arg-type]


def test_tiny_model_learns_and_writes_the_run_directory(tmp_path: Path) -> None:
    summary = train(tiny_cfg(tmp_path, run_name="learn"))
    run = tmp_path / "runs" / "learn"
    assert summary["status"] == "ok" and summary["final_step"] == 30
    loss = losses(run)
    assert loss[29] < loss[0] - 1.0  # the repeating pattern is learnable
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "ok" and manifest["world"] == 1 and manifest["grad_accum"] == 2
    assert manifest["peak_flops"] is None  # no datasheet peak for a laptop CPU: no MFU claimed
    vals = [r for r in read_log(run) if r["event"] == "val"]
    assert [v["step"] for v in vals] == [0, 10, 20, 30]
    steps = [r for r in read_log(run) if r["event"] == "step"]
    assert steps[0]["timing_warmup"] and not steps[5]["timing_warmup"]
    assert json.loads((run / "config.json").read_text())["preset"] == "tiny"


def test_checkpoint_resume_reproduces_the_uninterrupted_run(tmp_path: Path) -> None:
    straight = train(tiny_cfg(tmp_path, run_name="straight", max_steps=12, ckpt_every=6))
    assert straight["status"] == "ok"
    first = train(
        tiny_cfg(tmp_path, run_name="split", max_steps=12, ckpt_every=6, time_limit_s=1e-9)
    )
    assert first["status"] == "partial" and first["final_step"] == 1
    resumed = train(
        tiny_cfg(tmp_path, run_name="split", max_steps=12, ckpt_every=6, resume="latest")
    )
    assert resumed["status"] == "ok" and resumed["final_step"] == 12
    a = losses(tmp_path / "runs" / "straight")
    b = losses(tmp_path / "runs" / "split")
    assert set(a) == set(b)
    for step in a:
        assert b[step] == pytest.approx(a[step], rel=1e-5, abs=1e-6), step
    assert any(r["event"] == "resume" for r in read_log(tmp_path / "runs" / "split"))
    kept = sorted(p.name for p in (tmp_path / "runs" / "straight" / "ckpt").iterdir())
    assert kept == ["step_0000006", "step_0000012"]


def _torchrun(tmp_path: Path, name: str, parallel: str) -> Path:
    cfg = tiny_cfg(
        tmp_path, run_name=name, parallel=parallel, max_steps=6, val_every=0, val_steps=0
    )
    with socket.socket() as sock:  # a free port for a static loopback rendezvous (--standalone
        sock.bind(("127.0.0.1", 0))  # resolves localhost via IPv6 on macOS and can hang)
        port = sock.getsockname()[1]
    args = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--node_rank", "0",
        "--nproc_per_node", "2", "--master_addr", "127.0.0.1", "--master_port", str(port),
        "-m", "gptfsdp.cli", "train", "--set",
        *(f"{k}={v}" for k, v in cfg.__dict__.items() if v != "" and not isinstance(v, bool)),
        f"bf16={cfg.bf16}", f"compile={cfg.compile}",
    ]  # fmt: skip
    env = {
        **os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo",
    }  # fmt: skip
    out = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return tmp_path / "runs" / name


@pytest.mark.parametrize("parallel", ["ddp", "fsdp"])
def test_two_process_training_matches_one_process(tmp_path: Path, parallel: str) -> None:
    """2 gloo ranks x micro-batch 4 x 1 accumulation step == 1 rank x micro-batch 4 x 2 steps."""
    single = train(tiny_cfg(tmp_path, run_name="single", max_steps=6, val_every=0, val_steps=0))
    assert single["status"] == "ok"
    run = _torchrun(tmp_path, parallel, parallel)
    manifest = json.loads((run / "manifest.json").read_text())
    assert (
        manifest["world"] == 2 and manifest["parallel"] == parallel and manifest["grad_accum"] == 1
    )
    a, b = losses(tmp_path / "runs" / "single"), losses(run)
    for step in range(6):
        assert b[step] == pytest.approx(a[step], rel=2e-4), (parallel, step)
