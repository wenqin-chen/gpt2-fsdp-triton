"""Token shards and the distributed, resumable data loader.

A shard is a 1-D ``uint16`` NumPy array saved with ``np.save`` (``<prefix>_<split>_<nnnnnn>.npy``);
GPT-2's vocabulary (50,257) fits in 16 bits. ``split`` is ``val`` for shard 0 of the corpus and
``train`` for the rest, so validation tokens are never trained on.

``TokenLoader`` walks the shards of one split in order. Each optimizer micro-step consumes
``B * T * world`` tokens from a global cursor; rank ``r`` takes the ``r``-th ``B * T`` slice (plus
one token for the shifted targets). The loader's whole state is ``{"shard", "pos"}`` - identical
on every rank - so a checkpoint restores it exactly and a resumed run sees the same tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SHARD_RE = re.compile(r"_(train|val)_(\d{6})\.npy$")
TOKEN_DTYPE = np.uint16


def shard_name(prefix: str, split: str, index: int) -> str:
    return f"{prefix}_{split}_{index:06d}.npy"


def write_shard(path: str | Path, tokens: Sequence[int] | np.ndarray) -> Path:
    arr = np.asarray(tokens)
    if arr.size and (arr.min() < 0 or arr.max() > np.iinfo(TOKEN_DTYPE).max):
        raise ValueError("token ids must fit in uint16")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr.astype(TOKEN_DTYPE, copy=False))
    return out


def list_shards(data_dir: str | Path, split: str) -> list[Path]:
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    shards = sorted(
        p for p in Path(data_dir).glob("*.npy") if (m := SHARD_RE.search(p.name)) and m[1] == split
    )
    if not shards:
        raise FileNotFoundError(f"no {split} shards (*_{split}_nnnnnn.npy) in {data_dir}")
    return shards


def open_shard(path: str | Path) -> np.ndarray:
    """Memory-mapped ``uint16`` view of a shard (batches are converted to int64 on demand)."""
    arr: np.ndarray = np.load(path, mmap_mode="r")
    if arr.dtype != TOKEN_DTYPE or arr.ndim != 1:
        raise ValueError(f"{path}: expected a 1-D uint16 array, got {arr.dtype} {arr.shape}")
    return arr


@dataclass
class LoaderState:
    shard: int = 0
    pos: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"shard": self.shard, "pos": self.pos}


class TokenLoader:
    """Sequential (B, T) batches from one split for rank ``rank`` of ``world``."""

    def __init__(
        self, data_dir: str | Path, split: str, batch: int, seq: int, rank: int = 0, world: int = 1
    ) -> None:
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} outside world {world}")
        self.shards = list_shards(data_dir, split)
        self.batch, self.seq, self.rank, self.world = batch, seq, rank, world
        self.state = LoaderState()
        self._tokens: np.ndarray | None = None
        self._loaded: int | None = None
        # every shard must hold at least one full global step, or the walk would stall
        need = batch * seq * world + 1
        for p in self.shards:
            n = np.load(p, mmap_mode="r").shape[0]
            if n < need:
                raise ValueError(f"shard {p.name} has {n} tokens; one step needs {need}")

    @property
    def tokens_per_step(self) -> int:
        return self.batch * self.seq * self.world

    def _shard(self) -> np.ndarray:
        if self._loaded != self.state.shard:
            self._tokens = open_shard(self.shards[self.state.shard])
            self._loaded = self.state.shard
        assert self._tokens is not None
        return self._tokens

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._shard()
        start = self.state.pos + self.rank * self.batch * self.seq
        chunk = torch.from_numpy(tokens[start : start + self.batch * self.seq + 1].astype(np.int64))
        x = chunk[:-1].view(self.batch, self.seq)
        y = chunk[1:].view(self.batch, self.seq)
        self.state.pos += self.tokens_per_step
        if self.state.pos + self.tokens_per_step + 1 > tokens.shape[0]:
            self.state = LoaderState((self.state.shard + 1) % len(self.shards), 0)
        return x, y

    def state_dict(self) -> dict[str, int]:
        return self.state.to_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        shard, pos = int(state["shard"]), int(state["pos"])
        if not 0 <= shard < len(self.shards):
            raise ValueError(f"shard {shard} outside the {len(self.shards)} shards of this split")
        self.state = LoaderState(shard, pos)

    def reset(self) -> None:
        self.state = LoaderState()


def synthetic_corpus(
    out_dir: str | Path,
    n_train_shards: int = 2,
    shard_tokens: int = 4096,
    vocab: int = 512,
    seed: int = 0,
    prefix: str = "synthetic",
) -> Path:
    """Learnable toy shards for tests: a noisy repeating pattern over ``vocab`` token ids."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, vocab, size=64)
    out = Path(out_dir)
    for split, indices in (("val", [0]), ("train", range(1, n_train_shards + 1))):
        for i in indices:
            reps = np.tile(base, shard_tokens // base.size + 1)[:shard_tokens]
            noise = rng.random(shard_tokens) < 0.05
            reps[noise] = rng.integers(0, vocab, size=int(noise.sum()))
            write_shard(out / shard_name(prefix, split, i), reps)
    return out


def iter_documents_tokens(docs: Iterable[str], eot: int, encode) -> Iterable[np.ndarray]:  # type: ignore[no-untyped-def]
    """``<|endoftext|>`` + tokens for every document (GPT-2 convention: EOT *before* the text)."""
    for text in docs:
        ids = encode(text)
        arr = np.empty(len(ids) + 1, dtype=TOKEN_DTYPE)
        arr[0] = eot
        arr[1:] = ids
        yield arr
