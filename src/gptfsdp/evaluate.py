"""Held-out loss on the validation shard, for a run's checkpoint or for OpenAI's GPT-2 124M.

``val_loss`` scores the first ``tokens`` tokens of the validation shard (shard 0, never trained on)
in non-overlapping windows of ``seq_len``: the same tokens for every model, so the numbers compare
directly. ``load_openai_gpt2`` maps the Hugging Face ``openai-community/gpt2`` safetensors
(``transformer.*`` names, Conv1D weights stored transposed) onto :class:`gptfsdp.model.GPT` with the
original 50,257-token vocabulary, so the baseline is scored with exactly this code path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gptfsdp.data import list_shards, open_shard
from gptfsdp.model import GPT, GPTConfig

GPT2_VOCAB = 50257
CONV1D = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


def hf_to_ours(hf: dict[str, torch.Tensor], n_layer: int) -> dict[str, torch.Tensor]:
    """Rename (and transpose Conv1D weights of) a Hugging Face GPT-2 state dict."""
    out: dict[str, torch.Tensor] = {}
    for key, value in hf.items():
        name = key.removeprefix("transformer.")
        parts = name.split(".")
        if parts[-2:] in (["attn", "bias"], ["attn", "masked_bias"]):  # causal-mask buffers
            continue  # (the real projection bias is attn.c_attn.bias)
        if any(name.endswith(suffix) for suffix in CONV1D):
            value = value.t()
        out[name] = value.contiguous()
    out.setdefault("lm_head.weight", out["wte.weight"])
    expected = {f"h.{i}.ln_1.weight" for i in range(n_layer)}
    if not expected <= set(out):
        raise KeyError("state dict does not look like a GPT-2 checkpoint")
    return out


def ours_to_hf(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of :func:`hf_to_ours` (tests round-trip through it)."""
    out: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name == "lm_head.weight":
            continue
        if any(name.endswith(suffix) for suffix in CONV1D):
            value = value.t()
        out[f"transformer.{name}"] = value.contiguous()
    return out


def load_openai_gpt2(path: str | Path, cfg: GPTConfig | None = None) -> GPT:
    """GPT-2 124M from ``model.safetensors`` (a file or the directory holding it)."""
    from safetensors.torch import load_file

    src = Path(path)
    if src.is_dir():
        src = src / "model.safetensors"
    cfg = cfg or GPTConfig(vocab_size=GPT2_VOCAB)
    model = GPT(cfg)
    model.load_state_dict(hf_to_ours(load_file(str(src)), cfg.n_layer))
    return model


@torch.no_grad()
def val_loss(
    model: torch.nn.Module,
    data_dir: str | Path,
    tokens: int,
    seq_len: int = 1024,
    batch: int = 8,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Mean next-token loss over the first ``tokens`` tokens of the validation shard."""
    arr = open_shard(list_shards(data_dir, "val")[0])
    n_windows = min(tokens, arr.shape[0] - 1) // seq_len
    if n_windows < 1:
        raise ValueError("the validation shard is shorter than one window")
    model.eval()
    total, count = 0.0, 0
    dev = torch.device(device)
    for start in range(0, n_windows, batch):
        rows = range(start, min(start + batch, n_windows))
        chunk = np.stack([arr[r * seq_len : r * seq_len + seq_len + 1] for r in rows])
        t = torch.from_numpy(chunk.astype(np.int64)).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, _ = model(t[:, :-1])
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1), reduction="sum"
        )
        total += float(loss)
        count += t[:, 1:].numel()
    mean = total / count
    return {"tokens": count, "val_loss": mean, "perplexity": math.exp(mean)}


def load_run_model(run_dir: str | Path, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    """A run's newest complete checkpoint in a single-process model (DCP reshards FSDP state)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    from gptfsdp import checkpoint as ckpt
    from gptfsdp.cli import build_config

    run = Path(run_dir)
    cfg = build_config(str(run / "config.json"), [])
    src = ckpt.latest(run / "ckpt")
    if src is None:
        raise FileNotFoundError(f"no complete checkpoint under {run / 'ckpt'}")
    model = GPT(cfg.model_config()).to(device)
    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=str(src))
    set_model_state_dict(model, state["model"])
    extra: dict[str, Any] = json.loads((src / "extra.json").read_text())
    extra.update(checkpoint=src.name, seq_len=cfg.seq_len)
    return model, extra


def evaluate_checkpoint(run_dir: str | Path, data_dir: str | Path, tokens: int) -> dict[str, Any]:
    """``val_loss`` of a run's newest checkpoint, appended to its log as ``event: heldout``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, extra = load_run_model(run_dir, device)
    result = val_loss(model, data_dir, tokens, seq_len=int(extra["seq_len"]), device=device)
    record = {
        "event": "heldout",
        "checkpoint": extra["checkpoint"],
        "step": extra["step"],
        **result,
    }
    with (Path(run_dir) / "log.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def download_gpt2(out_dir: str | Path) -> dict[str, Any]:
    """OpenAI GPT-2 124M ``model.safetensors`` (``openai-community/gpt2``) at a pinned revision."""
    import hashlib

    from huggingface_hub import HfApi, hf_hub_download

    repo = "openai-community/gpt2"
    revision = HfApi().model_info(repo).sha
    path = Path(
        hf_hub_download(repo, "model.safetensors", revision=revision, local_dir=str(out_dir))
    )
    return {
        "repo_id": repo,
        "path": str(path),
        "revision": revision,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def evaluate_openai_gpt2(
    weights: str | Path, data_dir: str | Path, tokens: int, out: str | Path
) -> dict[str, Any]:
    """The baseline row: OpenAI's released GPT-2 124M scored on the same validation tokens."""
    import hashlib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = Path(weights)
    file = src / "model.safetensors" if src.is_dir() else src
    model = load_openai_gpt2(file).to(device)
    result = val_loss(model, data_dir, tokens, device=device)
    record = {
        "event": "heldout",
        "model": "openai-community/gpt2 (GPT-2 124M, MIT)",
        "weights_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        **result,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
