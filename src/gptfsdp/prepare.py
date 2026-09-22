"""FineWeb-Edu → GPT-2 token shards (runs on the cluster; ~27 GB of parquet in, ~19 GB out).

``prepare_fineweb`` downloads the parquet files of one subset from the Hugging Face Hub, streams
the ``text`` column through ``tiktoken`` in a process pool, prefixes every document with
``<|endoftext|>`` and cuts the token stream into ``uint16`` shards of ``shard_tokens`` tokens.
The first shard is the validation shard. ``manifest.json`` records the dataset revision, the
parquet files, and for every shard its split, token count and sha256.

``shard_stream`` is the pure part (texts in, shards out) and is what the tests exercise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from gptfsdp.data import TOKEN_DTYPE, shard_name, write_shard

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBSETS = {"sample-10BT": "sample/10BT/*.parquet", "sample-100BT": "sample/100BT/*.parquet"}
PREFIX = "edufineweb"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shard_stream(
    docs: Iterable[np.ndarray], out_dir: str | Path, shard_tokens: int, prefix: str = PREFIX
) -> list[dict[str, Any]]:
    """Pack token arrays into shards (shard 0 = val); the last partial shard is kept."""
    out = Path(out_dir)
    buf = np.empty(shard_tokens, dtype=TOKEN_DTYPE)
    fill, index, shards = 0, 0, []

    def flush(n: int) -> None:
        nonlocal index
        split = "val" if index == 0 else "train"
        path = write_shard(out / shard_name(prefix, split, index), buf[:n])
        shards.append(
            {"file": path.name, "split": split, "tokens": int(n), "sha256": _sha256(path)}
        )
        index += 1

    for tokens in docs:
        pos = 0
        while pos < len(tokens):
            take = min(shard_tokens - fill, len(tokens) - pos)
            buf[fill : fill + take] = tokens[pos : pos + take]
            fill += take
            pos += take
            if fill == shard_tokens:
                flush(fill)
                fill = 0
    if fill:
        flush(fill)
    return shards


def _encoder() -> tuple[Callable[[str], list[int]], int]:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    return enc.encode_ordinary, enc.eot_token


def _tokenize(text: str) -> np.ndarray:
    encode, eot = _encoder()
    ids = encode(text)
    arr = np.empty(len(ids) + 1, dtype=TOKEN_DTYPE)
    arr[0] = eot
    arr[1:] = ids
    return arr


def _iter_texts(parquet_files: list[Path], max_docs: int) -> Iterator[str]:
    import pyarrow.parquet as pq

    n = 0
    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=1024):
            for text in batch.column(0).to_pylist():
                yield text
                n += 1
                if max_docs and n >= max_docs:
                    return


def prepare_fineweb(
    out: str | Path,
    subset: str = "sample-10BT",
    shard_tokens: int = 100_000_000,
    workers: int = 2,
    max_docs: int = 0,
) -> dict[str, Any]:
    from multiprocessing import Pool

    from huggingface_hub import HfApi, snapshot_download

    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {sorted(SUBSETS)}")
    out_dir = Path(out)
    raw = out_dir / "raw"
    revision = HfApi().dataset_info(REPO_ID).sha
    snapshot_download(
        REPO_ID, repo_type="dataset", revision=revision, allow_patterns=[SUBSETS[subset]],
        local_dir=raw,
    )  # fmt: skip
    files = sorted(raw.glob(SUBSETS[subset]))
    if not files:
        raise FileNotFoundError(f"no parquet files for {subset} under {raw}")
    with Pool(workers) as pool:
        docs = pool.imap(_tokenize, _iter_texts(files, max_docs), chunksize=64)
        shards = shard_stream(docs, out_dir, shard_tokens)
    manifest = {
        "repo_id": REPO_ID, "subset": subset, "revision": revision,
        "parquet_files": [p.name for p in files], "tokenizer": "tiktoken gpt2 (encode_ordinary)",
        "eot_before_each_document": True, "shard_tokens": shard_tokens, "max_docs": max_docs,
        "shards": shards, "total_tokens": sum(s["tokens"] for s in shards),
    }  # fmt: skip
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"out": str(out_dir), "n_shards": len(shards), "total_tokens": manifest["total_tokens"]}
