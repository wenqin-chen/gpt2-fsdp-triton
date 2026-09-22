"""Resume-safe checkpoints with ``torch.distributed.checkpoint`` (DCP).

One code path for a single process, DDP and FSDP2: ``get_state_dict`` returns canonical
(unwrapped) keys and, under FSDP2, sharded DTensors that every rank writes in parallel.
Besides model and optimizer the checkpoint stores the step and the data-loader cursor, which is
all a resumed run needs to see exactly the tokens an uninterrupted run would have seen (the model
has no dropout, so no RNG state matters).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as tdist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

EXTRA_KEYS = ("step", "loader_shard", "loader_pos", "tokens_seen")


def step_dir(ckpt_root: str | Path, step: int) -> Path:
    return Path(ckpt_root) / f"step_{step:07d}"


def save(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, extra: dict[str, int]
) -> Path:
    missing = set(EXTRA_KEYS) - set(extra)
    if missing:
        raise ValueError(f"checkpoint extra state misses {sorted(missing)}")
    model_sd, optim_sd = get_state_dict(model, optimizer)
    out = Path(path)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(out))
    if not tdist.is_initialized() or tdist.get_rank() == 0:
        (out / "extra.json").write_text(json.dumps(extra, sort_keys=True) + "\n")
    if tdist.is_initialized():
        tdist.barrier()
    return out


def load(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    src = Path(path)
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(src))  # fills in place
    set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)
    extra: dict[str, Any] = json.loads((src / "extra.json").read_text())
    return extra


def latest(ckpt_root: str | Path) -> Path | None:
    """Newest complete checkpoint (one whose ``extra.json`` was written)."""
    done = sorted(p for p in Path(ckpt_root).glob("step_*") if (p / "extra.json").is_file())
    return done[-1] if done else None


def prune(ckpt_root: str | Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` complete checkpoints (rank 0 only); returns removed."""
    if keep < 1 or (tdist.is_initialized() and tdist.get_rank() != 0):
        return []
    done = sorted(p for p in Path(ckpt_root).glob("step_*") if (p / "extra.json").is_file())
    removed = done[:-keep]
    for p in removed:
        shutil.rmtree(p)
    return removed
