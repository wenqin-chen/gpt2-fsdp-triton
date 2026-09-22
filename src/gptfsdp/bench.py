"""LayerNorm benchmark: PyTorch eager vs ``torch.compile`` vs the Triton kernel (CUDA only).

For each (rows M, width N) the three implementations are timed with ``triton.testing.do_bench``
(warm-up and repeats handled there, CUDA events, median milliseconds) for the forward pass and for
forward + backward. Effective bandwidth counts the unavoidable HBM traffic of the fused operation:
forward reads x and writes y (2·M·N elements); forward + backward adds reading x and dy and
writing dx (5·M·N elements in total); weights, biases and the per-row statistics are ignored.
Before timing, the Triton output and gradients are checked against the eager fp32 reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SHAPES: tuple[tuple[int, int], ...] = (
    (8192, 768), (16384, 768), (65536, 768), (16384, 1024), (16384, 2048), (16384, 4096),
)  # fmt: skip


def _check(m: int, n: int, dtype: torch.dtype) -> dict[str, float]:
    from gptfsdp.kernels.layernorm import layer_norm

    torch.manual_seed(0)
    x = torch.randn(m, n, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn(n, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=dtype)
    y = layer_norm(x, w, b)
    y.backward(dy)
    grads = [t.grad.detach().clone() for t in (x, w, b)]  # type: ignore[union-attr]
    xr, wr, br = (t.detach().float().requires_grad_() for t in (x, w, b))
    yr = F.layer_norm(xr, (n,), wr, br, 1e-5)
    yr.backward(dy.float())
    errs = {"y": (y.float() - yr).abs().max().item()}
    for name, g, ref in zip(("dx", "dw", "db"), grads, (xr.grad, wr.grad, br.grad), strict=True):
        assert ref is not None
        errs[name] = (g.float() - ref).abs().max().item()
    return errs


def bench_layernorm(out: str | Path, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("the LayerNorm benchmark needs a CUDA GPU")
    from triton.testing import do_bench

    from gptfsdp.kernels.layernorm import layer_norm

    compiled = torch.compile(lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, 1e-5))
    impls = {
        "eager": lambda x, w, b: F.layer_norm(x, (x.shape[-1],), w, b, 1e-5),
        "compile": compiled,
        "triton": layer_norm,
    }
    rows: list[dict[str, Any]] = []
    elem = torch.tensor([], dtype=dtype).element_size()
    for m, n in SHAPES:
        errors = _check(m, n, dtype)
        x = torch.randn(m, n, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.ones(n, device="cuda", dtype=dtype, requires_grad=True)
        b = torch.zeros(n, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn(m, n, device="cuda", dtype=dtype)
        row: dict[str, Any] = {
            "M": m,
            "N": n,
            "dtype": str(dtype).removeprefix("torch."),
            "max_abs_err": errors,
        }
        for name, fn in impls.items():

            def fwd(fn=fn, x=x, w=w, b=b):  # type: ignore[no-untyped-def]
                return fn(x, w, b)

            def fwd_bwd(fn=fn, x=x, w=w, b=b, dy=dy):  # type: ignore[no-untyped-def]
                fn(x, w, b).backward(dy)

            ms_f = float(do_bench(fwd, warmup=25, rep=100))
            ms_fb = float(do_bench(fwd_bwd, warmup=25, rep=100, grad_to_none=[x, w, b]))
            row[name] = {
                "fwd_ms": ms_f,
                "fwd_bwd_ms": ms_fb,
                "fwd_gbps": 2 * m * n * elem / ms_f / 1e6,
                "fwd_bwd_gbps": 5 * m * n * elem / ms_fb / 1e6,
            }
        for base in ("eager", "compile"):
            row[f"speedup_fwd_vs_{base}"] = row[base]["fwd_ms"] / row["triton"]["fwd_ms"]
            row[f"speedup_fwd_bwd_vs_{base}"] = (
                row[base]["fwd_bwd_ms"] / row["triton"]["fwd_bwd_ms"]
            )
        rows.append(row)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "rows": rows,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"out": str(path), "device": result["device"], "n_shapes": len(rows)}
