"""Learning-rate schedule and throughput / MFU accounting.

MFU = achieved model FLOP/s / (n_gpus * peak dense bf16 FLOP/s). The achieved rate is
``flops_per_token * tokens_per_s`` with nanoGPT's per-token count (see :mod:`gptfsdp.model`).
Peak values are NVIDIA datasheet numbers *without* sparsity (the datasheets quote the sparse
figure, twice the dense one); every run logs the peak it used and where it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# dense bf16 tensor-core peak, FLOP/s, by marketing name (longest match wins)
PEAK_BF16_DENSE: dict[str, tuple[float, str]] = {
    "H200 NVL": (835e12, "NVIDIA H200 NVL datasheet: 1,671 TFLOPS with sparsity / 2"),
    "H200": (989.5e12, "NVIDIA H200 SXM datasheet: 1,979 TFLOPS with sparsity / 2"),
    "H100 PCIe": (756e12, "NVIDIA H100 PCIe datasheet: 1,513 TFLOPS with sparsity / 2"),
    "H100": (989.5e12, "NVIDIA H100 SXM datasheet: 1,979 TFLOPS with sparsity / 2"),
    "A100": (312e12, "NVIDIA A100 datasheet: 312 TFLOPS dense BF16"),
}


@dataclass(frozen=True)
class Peak:
    flops: float | None
    source: str


def peak_for(device_name: str) -> Peak:
    for key in sorted(PEAK_BF16_DENSE, key=len, reverse=True):
        if key in device_name:
            flops, source = PEAK_BF16_DENSE[key]
            return Peak(flops, source)
    return Peak(None, f"no datasheet peak for {device_name!r}; MFU not reported")


def mfu(flops_per_token: float, tokens_per_s: float, n_gpus: int, peak: Peak) -> float | None:
    if peak.flops is None or n_gpus < 1:
        return None
    return flops_per_token * tokens_per_s / (n_gpus * peak.flops)


@dataclass(frozen=True)
class CosineSchedule:
    """Linear warm-up to ``max_lr`` over ``warmup`` steps, cosine decay to ``min_lr`` at
    ``max_steps``, constant afterwards."""

    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup: int = 715
    max_steps: int = 19073

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.max_lr * (step + 1) / self.warmup
        if step >= self.max_steps:
            return self.min_lr
        ratio = (step - self.warmup) / max(1, self.max_steps - self.warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
