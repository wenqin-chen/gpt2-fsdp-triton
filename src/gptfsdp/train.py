"""The training loop (single process, DDP or FSDP2) and its run directory.

``runs/<run_id>/`` holds ``config.json``, ``manifest.json`` (git sha, torch/CUDA versions, device,
world size, MFU peak and its source, start/end, status), ``log.jsonl`` (one JSON object per
logged step, validation pass and evaluation) and ``ckpt/step_nnnnnnn/`` (DCP checkpoints).
Every number that reaches RESULTS.md is read back from these files.

Throughput is measured per optimizer step with a device synchronisation at the end of the step;
``tokens_per_s = tokens_per_step / dt``. The first ``cfg.warmup_timing_steps`` steps (compilation,
allocator warm-up) are logged but flagged ``timing_warmup``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from gptfsdp import checkpoint as ckpt
from gptfsdp import dist
from gptfsdp.data import TokenLoader
from gptfsdp.model import GPT, GPTConfig
from gptfsdp.perf import CosineSchedule, mfu, peak_for

PRESETS: dict[str, dict[str, int]] = {
    "gpt2-small": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "tiny": {"n_layer": 2, "n_head": 2, "n_embd": 64},  # tests
}


@dataclass
class TrainConfig:
    data_dir: str = "data/fineweb_edu_10b"
    out_dir: str = "runs"
    run_name: str = ""
    # model
    preset: str = "gpt2-small"
    vocab_size: int = 50304
    seq_len: int = 1024
    norm: str = "torch"
    activation_checkpointing: bool = False
    # batch: tokens per optimizer step = micro_batch * seq_len * world * grad_accum
    total_batch_tokens: int = 524288
    micro_batch: int = 16
    # optimisation
    max_steps: int = 19073
    warmup_steps: int = 715
    max_lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # system
    parallel: str = "none"
    device: str = "auto"
    bf16: bool = True
    compile: bool = False
    seed: int = 1337
    # logging, evaluation, checkpoints
    log_every: int = 1
    warmup_timing_steps: int = 10
    val_every: int = 250
    val_steps: int = 20
    ckpt_every: int = 1000
    ckpt_keep: int = 2
    resume: str = ""  # a checkpoint directory, or "latest" (this run's newest)
    time_limit_s: float = 0.0  # > 0: checkpoint and stop before this wall-clock budget runs out

    def model_config(self) -> GPTConfig:
        dims = PRESETS[self.preset]
        return GPTConfig(
            block_size=self.seq_len, vocab_size=self.vocab_size, norm=self.norm,
            activation_checkpointing=self.activation_checkpointing, **dims,
        )  # fmt: skip


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )  # fmt: skip
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunLog:
    """Rank-0 JSONL log plus the manifest; other ranks write nothing."""

    def __init__(self, run_dir: Path, enabled: bool) -> None:
        self.run_dir, self.enabled = run_dir, enabled
        if enabled:
            run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        if self.enabled:
            with (self.run_dir / "log.jsonl").open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def manifest(self, payload: dict[str, Any]) -> None:
        if self.enabled:
            path = self.run_dir / "manifest.json"
            old = json.loads(path.read_text()) if path.is_file() else {}
            path.write_text(json.dumps({**old, **payload}, indent=2, sort_keys=True) + "\n")


def evaluate_val(
    model: Any, loader: TokenLoader, steps: int, ctx: dist.DistContext, autocast: Any
) -> float:
    model.eval()
    loader.reset()
    total = torch.zeros((), device=ctx.device)
    with torch.no_grad():
        for _ in range(steps):
            x, y = loader.next_batch()
            with autocast():
                _, loss = model(x.to(ctx.device), y.to(ctx.device))
            total += loss.detach().float()
    model.train()
    return float(dist.all_reduce_mean(total / steps, ctx).item())


def train(cfg: TrainConfig) -> dict[str, Any]:
    ctx = dist.setup(cfg.device)
    torch.manual_seed(cfg.seed)
    if ctx.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_id = cfg.run_name or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = Path(cfg.out_dir) / run_id
    log = RunLog(run_dir, ctx.is_main)
    if ctx.is_main:
        (run_dir / "config.json").write_text(
            json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True) + "\n"
        )

    tokens_per_micro = cfg.micro_batch * cfg.seq_len * ctx.world
    if cfg.total_batch_tokens % tokens_per_micro:
        raise ValueError(
            f"total_batch_tokens {cfg.total_batch_tokens} is not a multiple of micro_batch x "
            f"seq_len x world = {tokens_per_micro}"
        )
    grad_accum = cfg.total_batch_tokens // tokens_per_micro

    mcfg = cfg.model_config()
    model = GPT(mcfg).to(ctx.device)
    flops_per_token = model.flops_per_token()
    n_params = model.num_params(non_embedding=False)
    use_bf16 = cfg.bf16 and ctx.device.type == "cuda"
    wrapped = dist.wrap(model, cfg.parallel, ctx, bf16=use_bf16)
    optimizer = dist.unwrap(wrapped).configure_optimizer(  # type: ignore[operator]
        cfg.weight_decay, cfg.max_lr, (0.9, 0.95), ctx.device.type
    )
    step_model = torch.compile(wrapped) if cfg.compile else wrapped

    # FSDP's mixed-precision policy already computes in bf16; DDP / single use autocast
    def autocast():  # type: ignore[no-untyped-def]
        if use_bf16 and cfg.parallel != "fsdp":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type=ctx.device.type, enabled=False)

    train_loader = TokenLoader(
        cfg.data_dir, "train", cfg.micro_batch, cfg.seq_len, ctx.rank, ctx.world
    )
    val_loader = TokenLoader(cfg.data_dir, "val", cfg.micro_batch, cfg.seq_len, ctx.rank, ctx.world)
    schedule = CosineSchedule(
        max_lr=cfg.max_lr, min_lr=cfg.max_lr * cfg.min_lr_ratio, warmup=cfg.warmup_steps,
        max_steps=cfg.max_steps,
    )  # fmt: skip

    start_step, tokens_seen = 0, 0
    ckpt_root = run_dir / "ckpt"
    if cfg.resume:
        src = ckpt.latest(ckpt_root) if cfg.resume == "latest" else Path(cfg.resume)
        if src is not None:
            extra = ckpt.load(src, wrapped, optimizer)
            start_step, tokens_seen = int(extra["step"]), int(extra["tokens_seen"])
            train_loader.load_state_dict(
                {"shard": int(extra["loader_shard"]), "pos": int(extra["loader_pos"])}
            )
            log.write({"event": "resume", "from": str(src), "step": start_step})

    device_name = (
        torch.cuda.get_device_name(ctx.device)
        if ctx.device.type == "cuda"
        else platform.processor() or "cpu"
    )
    peak = peak_for(device_name)
    log.manifest({
        "run_id": run_id, "status": "running", "started": _now(), "git_sha": _git_sha(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": device_name,
        "world": ctx.world, "parallel": cfg.parallel, "n_params": n_params,
        "flops_per_token": flops_per_token, "peak_flops": peak.flops, "peak_source": peak.source,
        "grad_accum": grad_accum, "tokens_per_step": cfg.total_batch_tokens,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })  # fmt: skip

    t_start = time.perf_counter()
    status, step = "ok", start_step
    wrapped.train()
    for step in range(start_step, cfg.max_steps):
        if cfg.val_every and step % cfg.val_every == 0:
            val = evaluate_val(step_model, val_loader, cfg.val_steps, ctx, autocast)
            log.write({"event": "val", "step": step, "val_loss": val, "tokens_seen": tokens_seen})
        t0 = time.perf_counter()
        loss_accum = torch.zeros((), device=ctx.device)
        for micro in range(grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(ctx.device, non_blocking=True), y.to(ctx.device, non_blocking=True)
            with dist.gradient_sync(wrapped, cfg.parallel, micro == grad_accum - 1):
                with autocast():
                    _, loss = step_model(x, y)
                loss = loss / grad_accum
                loss.backward()
            loss_accum += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), cfg.grad_clip)
        if hasattr(norm, "full_tensor"):  # DTensor under FSDP2
            norm = norm.full_tensor()
        lr = schedule(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        dt = time.perf_counter() - t0
        tokens_seen += cfg.total_batch_tokens
        loss_value = float(dist.all_reduce_mean(loss_accum, ctx).item())
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            tps = cfg.total_batch_tokens / dt
            record: dict[str, Any] = {
                "event": "step", "step": step, "loss": loss_value, "lr": lr,
                "grad_norm": float(norm), "dt_s": dt, "tokens_per_s": tps,
                "mfu": mfu(flops_per_token, tps, ctx.world, peak), "tokens_seen": tokens_seen,
                "timing_warmup": step - start_step < cfg.warmup_timing_steps,
            }  # fmt: skip
            if ctx.device.type == "cuda":
                record["max_mem_gb"] = torch.cuda.max_memory_allocated(ctx.device) / 2**30
            log.write(record)
        done = step == cfg.max_steps - 1
        out_of_time = cfg.time_limit_s > 0 and time.perf_counter() - t_start > cfg.time_limit_s
        if (cfg.ckpt_every and (step + 1) % cfg.ckpt_every == 0) or done or out_of_time:
            ckpt.save(
                ckpt.step_dir(ckpt_root, step + 1), wrapped, optimizer,
                {"step": step + 1, "loader_shard": train_loader.state.shard,
                 "loader_pos": train_loader.state.pos, "tokens_seen": tokens_seen},
            )  # fmt: skip
            ckpt.prune(ckpt_root, cfg.ckpt_keep)
        if out_of_time and not done:
            status = "partial"
            break

    final_val = (
        evaluate_val(step_model, val_loader, cfg.val_steps, ctx, autocast)
        if cfg.val_steps
        else None
    )
    if final_val is not None:
        log.write(
            {"event": "val", "step": step + 1, "val_loss": final_val, "tokens_seen": tokens_seen}
        )
    summary = {
        "run_id": run_id, "status": status, "steps_done": step + 1 - start_step,
        "final_step": step + 1, "tokens_seen": tokens_seen, "final_val_loss": final_val,
        "wall_s": time.perf_counter() - t_start,
    }  # fmt: skip
    log.manifest({**summary, "finished": _now()})
    dist.teardown(ctx)
    return summary
