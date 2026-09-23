"""Fused LayerNorm in Triton: forward and backward (dx, dweight, dbias).

Forward: one program per row; statistics in fp32; writes ``y`` in the input dtype and the per-row
``mean`` and ``rstd`` for the backward pass. (A 2-D tiled forward and a fused dW/dB reduction kernel
were tried and measured slower on an H200; see RESULTS.md.)

Backward: each program owns ``ROWS_PER_PROG`` consecutive rows and walks them in 2-D tiles of
``ROWS`` rows x ``BLOCK_N`` columns (several rows in flight per iteration); for every row

    xhat = (x - mean) * rstd,   g = w * dy,
    dx   = (g - (xhat * mean(xhat * g) + mean(g))) * rstd,

and ``dy * xhat`` and ``dy`` are accumulated into fp32 partial sums that are written once per
program and reduced with ``torch.sum`` (no atomics, no locks). The launch uses ~8 programs per SM;
``ROWS`` shrinks as rows get wider to keep the tile in registers. (v1 walked one row at a time
with ~4 programs per SM and was latency-bound: slower than eager for GPT-2 shapes.)

``TritonLayerNorm`` has the same parameters (``weight``, ``bias``) and state-dict keys as
``nn.LayerNorm``, so checkpoints are interchangeable; on CPU tensors it falls back to
``F.layer_norm`` (the laptop tests exercise the model, the GPU tests the kernel).
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Triton ships with CUDA builds of PyTorch on Linux only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on machines without Triton
    HAS_TRITON = False

MAX_FUSED_BYTES = 65536

if HAS_TRITON:

    @triton.jit
    def _ln_fwd(
        X, Y, W, B, Mean, Rstd,
        stride, N, eps,
        BLOCK_N: tl.constexpr,
    ):  # fmt: skip
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = xc * rstd * w + b
        tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)
        tl.store(Mean + row, mean)
        tl.store(Rstd + row, rstd)

    @triton.jit
    def _ln_bwd(
        DY, X, W, Mean, Rstd, DX, DW_part, DB_part,
        stride, N, M,
        ROWS_PER_PROG: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr,
    ):  # fmt: skip
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        col_mask = cols < N
        w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
        dw_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        db_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for r0 in range(0, ROWS_PER_PROG, ROWS):
            rows = pid * ROWS_PER_PROG + r0 + tl.arange(0, ROWS)
            row_mask = rows < M
            mask = row_mask[:, None] & col_mask[None, :]
            offs = rows[:, None] * stride + cols[None, :]
            x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(DY + offs, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask=row_mask, other=0.0)
            rstd = tl.load(Rstd + rows, mask=row_mask, other=0.0)
            xhat = tl.where(mask, (x - mean[:, None]) * rstd[:, None], 0.0)
            g = tl.where(mask, w[None, :] * dy, 0.0)
            c1 = tl.sum(xhat * g, axis=1) / N
            c2 = tl.sum(g, axis=1) / N
            dx = (g - (xhat * c1[:, None] + c2[:, None])) * rstd[:, None]
            tl.store(DX + offs, dx.to(DX.dtype.element_ty), mask=mask)
            dw_acc += tl.sum(dy * xhat, axis=0)
            db_acc += tl.sum(dy, axis=0)
        tl.store(DW_part + pid * N + cols, dw_acc, mask=col_mask)
        tl.store(DB_part + pid * N + cols, db_acc, mask=col_mask)


@functools.lru_cache(maxsize=16)
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _bwd_params(m: int, block_n: int, device_index: int) -> tuple[int, int, int, int]:
    """(ROWS, ROWS_PER_PROG, n_prog, num_warps): ~8 programs per SM, tiles of <= 4096 elements."""
    rows = max(1, min(4, 4096 // block_n))
    per_prog = triton.cdiv(triton.cdiv(m, 8 * _sm_count(device_index)), rows) * rows
    n_prog = triton.cdiv(m, per_prog)
    num_warps = min(max(rows * block_n // 512, 1), 8)
    return rows, per_prog, n_prog, num_warps


def _launch_params(n: int, element_size: int) -> tuple[int, int]:
    block_n = triton.next_power_of_2(n)
    if block_n * element_size > MAX_FUSED_BYTES:
        raise ValueError(f"row of {n} elements does not fit one fused block")
    num_warps = min(max(block_n // 256, 1), 8)
    return block_n, num_warps


class _LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:  # type: ignore[override]
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        m, n = x2.shape
        block_n, num_warps = _launch_params(n, x2.element_size())
        y = torch.empty_like(x2)
        mean = torch.empty(m, dtype=torch.float32, device=x.device)
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)
        _ln_fwd[(m,)](
            x2, y, weight, bias, mean, rstd, x2.stride(0), n, eps,
            BLOCK_N=block_n, num_warps=num_warps,
        )  # fmt: skip
        ctx.save_for_backward(x2, weight, mean, rstd)
        ctx.shape, ctx.block_n = shape, block_n
        ctx.bias_dtype = bias.dtype
        return y.view(shape)

    @staticmethod
    def backward(
        ctx: Any, dy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:  # type: ignore[override]
        x2, weight, mean, rstd = ctx.saved_tensors
        m, n = x2.shape
        dy2 = dy.reshape(-1, n).contiguous()
        dx = torch.empty_like(x2)
        rows, per_prog, n_prog, num_warps = _bwd_params(m, ctx.block_n, x2.device.index or 0)
        dw_part = torch.empty((n_prog, n), dtype=torch.float32, device=x2.device)
        db_part = torch.empty((n_prog, n), dtype=torch.float32, device=x2.device)
        _ln_bwd[(n_prog,)](
            dy2, x2, weight, mean, rstd, dx, dw_part, db_part, x2.stride(0), n, m,
            ROWS_PER_PROG=per_prog, ROWS=rows, BLOCK_N=ctx.block_n, num_warps=num_warps,
        )  # fmt: skip
        dw = dw_part.sum(0).to(weight.dtype)
        db = db_part.sum(0).to(ctx.bias_dtype)
        return dx.view(ctx.shape), dw, db, None


def layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Fused LayerNorm over the last dimension (CUDA + Triton), ``F.layer_norm`` otherwise."""
    if not (HAS_TRITON and x.is_cuda):
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    out: torch.Tensor = _LayerNormFn.apply(x, weight, bias, eps)
    return out


class TritonLayerNorm(nn.Module):
    """Drop-in for ``nn.LayerNorm(n)`` (elementwise affine, eps 1e-5)."""

    def __init__(self, n: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (n,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return layer_norm(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{self.normalized_shape[0]}, eps={self.eps}, triton={HAS_TRITON}"
