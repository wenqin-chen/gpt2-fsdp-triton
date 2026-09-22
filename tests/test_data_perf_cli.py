from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gptfsdp.cli import build_config
from gptfsdp.data import (
    TokenLoader,
    list_shards,
    shard_name,
    synthetic_corpus,
    write_shard,
)
from gptfsdp.perf import CosineSchedule, Peak, mfu, peak_for
from gptfsdp.prepare import shard_stream


def _corpus(tmp_path: Path, n: int = 1000) -> Path:
    for split, i in (("val", 0), ("train", 1), ("train", 2)):
        write_shard(tmp_path / shard_name("t", split, i), np.arange(i * n, (i + 1) * n) % 60000)
    return tmp_path


def test_shards_and_split_listing(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert [p.name for p in list_shards(tmp_path, "train")] == [
        "t_train_000001.npy",
        "t_train_000002.npy",
    ]
    assert [p.name for p in list_shards(tmp_path, "val")] == ["t_val_000000.npy"]
    with pytest.raises(FileNotFoundError):
        list_shards(tmp_path / "empty", "train")
    with pytest.raises(ValueError, match="uint16"):
        write_shard(tmp_path / "bad.npy", [70000])


def test_loader_targets_ranks_and_rollover(tmp_path: Path) -> None:
    _corpus(tmp_path)
    loaders = [TokenLoader(tmp_path, "train", 2, 8, rank=r, world=2) for r in range(2)]
    batches = [ld.next_batch() for ld in loaders]
    (x0, y0), (x1, y1) = batches
    assert torch.equal(y0[:, :-1], x0[:, 1:])  # targets are the shifted inputs
    start = 1000  # shard 1 starts at token id 1000
    assert x0[0, 0].item() == start and x1[0, 0].item() == start + 16  # disjoint rank slices
    # walk until the shard is exhausted: the cursor moves on to shard 2, then wraps to shard 1
    seen_shards = set()
    for _ in range(200):
        seen_shards.add(loaders[0].state.shard)
        loaders[0].next_batch()
    assert seen_shards == {0, 1}


def test_loader_state_round_trip(tmp_path: Path) -> None:
    _corpus(tmp_path)
    a = TokenLoader(tmp_path, "train", 2, 8)
    for _ in range(5):
        a.next_batch()
    b = TokenLoader(tmp_path, "train", 2, 8)
    b.load_state_dict(a.state_dict())
    xa, _ = a.next_batch()
    xb, _ = b.next_batch()
    assert torch.equal(xa, xb)
    with pytest.raises(ValueError, match="outside"):
        b.load_state_dict({"shard": 9, "pos": 0})
    with pytest.raises(ValueError, match="one step needs"):
        TokenLoader(tmp_path, "train", 64, 64)


def test_synthetic_corpus_layout(tmp_path: Path) -> None:
    synthetic_corpus(tmp_path, n_train_shards=3, shard_tokens=512, vocab=100)
    assert len(list_shards(tmp_path, "train")) == 3 and len(list_shards(tmp_path, "val")) == 1
    arr = np.load(list_shards(tmp_path, "val")[0])
    assert arr.dtype == np.uint16 and arr.shape == (512,) and arr.max() < 100


def test_shard_stream_packs_documents(tmp_path: Path) -> None:
    docs = [np.array([50256, 1, 2, 3], dtype=np.uint16), np.array([50256, 4, 5], dtype=np.uint16)]
    shards = shard_stream(docs, tmp_path, shard_tokens=3, prefix="x")
    assert [s["file"] for s in shards] == [
        "x_val_000000.npy",
        "x_train_000001.npy",
        "x_train_000002.npy",
    ]
    assert [s["tokens"] for s in shards] == [3, 3, 1]
    stream = np.concatenate([np.load(tmp_path / s["file"]) for s in shards])
    assert stream.tolist() == [50256, 1, 2, 3, 50256, 4, 5]
    assert all(len(s["sha256"]) == 64 for s in shards)


def test_cosine_schedule() -> None:
    s = CosineSchedule(max_lr=1.0, min_lr=0.1, warmup=10, max_steps=110)
    assert s(0) == pytest.approx(0.1) and s(9) == pytest.approx(1.0)
    assert s(60) == pytest.approx(0.55)  # half-way through the cosine
    assert s(110) == pytest.approx(0.1) and s(500) == pytest.approx(0.1)


def test_mfu_and_peaks() -> None:
    assert peak_for("NVIDIA H200").flops == pytest.approx(989.5e12)
    assert peak_for("NVIDIA H200 NVL").flops == pytest.approx(835e12)
    assert peak_for("NVIDIA A100-SXM4-80GB").flops == pytest.approx(312e12)
    unknown = peak_for("Apple M2 Pro")
    assert unknown.flops is None and mfu(1e9, 1e5, 1, unknown) is None
    assert mfu(1e9, 5e5, 8, Peak(1e15, "x")) == pytest.approx(1e9 * 5e5 / 8e15)


def test_build_config_coerces_and_rejects(tmp_path: Path) -> None:
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text('{"preset": "tiny", "max_steps": 5}')
    cfg = build_config(str(cfg_file), ["micro_batch=4", "bf16=false", "max_lr=0.001"])
    assert cfg.preset == "tiny" and cfg.max_steps == 5 and cfg.micro_batch == 4
    assert cfg.bf16 is False and cfg.max_lr == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="key=value"):
        build_config(None, ["nonsense=1"])
    cfg_file.write_text('{"typo_key": 1}')
    with pytest.raises(ValueError, match="unknown config keys"):
        build_config(str(cfg_file), [])
