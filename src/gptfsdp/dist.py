"""Process groups and model wrapping: single process, DDP or FSDP2 (``fully_shard``).

``setup()`` reads the ``torchrun`` environment (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``); NCCL on
CUDA, gloo on CPU (tests run two gloo processes on a laptop). ``wrap()`` returns the module to
train: DDP around the model, or the model itself after ``fully_shard`` has been applied to every
transformer block and then to the root (FSDP2 shards parameters as DTensors in place).
FSDP's mixed-precision policy keeps fp32 master weights, computes in bf16 and reduces
gradients in fp32.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as tdist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

PARALLEL_MODES = ("none", "ddp", "fsdp")


@dataclass(frozen=True)
class DistContext:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: str | None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.backend is not None


def setup(device: str = "auto") -> DistContext:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available() and device in ("auto", "cuda")
    if use_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    backend = None
    if world > 1:
        backend = "nccl" if use_cuda else "gloo"
        if not tdist.is_initialized():
            tdist.init_process_group(backend=backend, device_id=dev if use_cuda else None)
    return DistContext(rank, world, local_rank, dev, backend)


def teardown(ctx: DistContext) -> None:
    if ctx.distributed and tdist.is_initialized():
        tdist.barrier()
        tdist.destroy_process_group()


def _fully_shard():  # type: ignore[no-untyped-def]
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import (  # type: ignore[no-redef]
            MixedPrecisionPolicy,
            fully_shard,
        )
    return fully_shard, MixedPrecisionPolicy


def wrap(model: nn.Module, mode: str, ctx: DistContext, bf16: bool) -> nn.Module:
    if mode not in PARALLEL_MODES:
        raise ValueError(f"parallel mode must be one of {PARALLEL_MODES}, got {mode!r}")
    if mode == "none":
        return model
    if not ctx.distributed:
        raise ValueError(f"parallel={mode!r} needs torchrun (WORLD_SIZE > 1)")
    if mode == "ddp":
        ids = [ctx.local_rank] if ctx.device.type == "cuda" else None
        return DDP(model, device_ids=ids)
    from torch.distributed.device_mesh import init_device_mesh

    fully_shard, policy_cls = _fully_shard()
    mesh = init_device_mesh(ctx.device.type, (ctx.world,))
    policy = (
        policy_cls(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if bf16 else policy_cls()
    )
    for block in model.h:  # type: ignore[union-attr]
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    """The module that owns the parameters (FSDP2 shards in place; DDP wraps)."""
    return model.module if isinstance(model, DDP) else model


@contextlib.contextmanager
def gradient_sync(model: nn.Module, mode: str, enabled: bool) -> Iterator[None]:
    """Skip the gradient all-reduce / reduce-scatter on all but the last micro-step."""
    if mode == "ddp" and not enabled:
        with model.no_sync():  # type: ignore[operator]
            yield
        return
    if mode == "fsdp":
        model.set_requires_gradient_sync(enabled)  # type: ignore[operator]
    yield


def all_reduce_mean(value: torch.Tensor, ctx: DistContext) -> torch.Tensor:
    if ctx.distributed:
        tdist.all_reduce(
            value, op=tdist.ReduceOp.AVG if ctx.backend == "nccl" else tdist.ReduceOp.SUM
        )
        if ctx.backend != "nccl":  # gloo has no AVG
            value /= ctx.world
    return value
