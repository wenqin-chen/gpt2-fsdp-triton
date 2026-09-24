"""Multiple-choice completion scoring (HellaSwag format; used for ARC-Easy, see :mod:`gptfsdp.arc`).

HellaSwag itself (validation, 10,042 items) was dropped: its upstream repository is DMCA-blocked.

Each item has a context and four endings. Every ending is scored by the model's cross-entropy on
the ending tokens given the context; the prediction is the ending with the lowest *mean* token
loss (length-normalised, reported as ``acc_norm``) and, for reference, the lowest *summed* loss
(``acc``). Endings are encoded with a leading space, as GPT-2 would see them after the context.
Items are split across ranks (``i % world == rank``) and the counts all-reduced.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

URL = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
N_VAL = 10042


def iter_items(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def render(
    item: dict[str, Any], encode: Callable[[str], list[int]]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(tokens [k, L], ending mask [k, L], label) for k endings, right-padded with zeros."""
    ctx = encode(item["ctx"])
    rows, masks = [], []
    for ending in item["endings"]:
        end = encode(" " + ending)
        rows.append(ctx + end)
        masks.append([0] * len(ctx) + [1] * len(end))
    width = max(len(r) for r in rows)
    tokens = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks, strict=True)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, int(item["label"])


def predict(logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[int, int]:
    """(argmin summed loss, argmin mean loss) over the four endings."""
    shift_logits = logits[:, :-1, :].float()
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:].float()
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_tokens.reshape(-1), reduction="none"
    ).view(tokens.size(0), -1)
    summed = (losses * shift_mask).sum(dim=1)
    mean = summed / shift_mask.sum(dim=1)
    return int(summed.argmin().item()), int(mean.argmin().item())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    items: Iterable[dict[str, Any]],
    encode: Callable[[str], list[int]],
    device: torch.device,
    rank: int = 0,
    world: int = 1,
    autocast: Callable[[], Any] | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    counts = torch.zeros(3, dtype=torch.long, device=device)  # n, correct_sum, correct_norm
    for i, item in enumerate(items):
        if i % world != rank:
            continue
        tokens, mask, label = render(item, encode)
        tokens, mask = tokens.to(device), mask.to(device)
        if autocast is not None:
            with autocast():
                logits, _ = model(tokens)
        else:
            logits, _ = model(tokens)
        pred_sum, pred_norm = predict(logits, tokens, mask)
        counts += torch.tensor([1, pred_sum == label, pred_norm == label], device=device)
    import torch.distributed as tdist

    if world > 1 and tdist.is_initialized():
        tdist.all_reduce(counts)
    model.train(was_training)
    n = int(counts[0])
    return {
        "n": n,
        "acc": int(counts[1]) / max(1, n),
        "acc_norm": int(counts[2]) / max(1, n),
    }


def download(path: str | Path, url: str = URL) -> Path:
    import httpx

    out = Path(path)
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with out.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return out


def evaluate_run(run_dir: str | Path, data: str | Path, limit: int = 0) -> dict[str, Any]:
    """HellaSwag of a run's newest checkpoint on one device (DCP reshards FSDP checkpoints)."""
    import tiktoken
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    from gptfsdp import checkpoint as ckpt
    from gptfsdp.cli import build_config
    from gptfsdp.model import GPT

    run = Path(run_dir)
    cfg = build_config(str(run / "config.json"), [])
    src = ckpt.latest(run / "ckpt")
    if src is None:
        raise FileNotFoundError(f"no complete checkpoint under {run / 'ckpt'}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(cfg.model_config()).to(device)
    # same key canonicalisation as checkpoint.save (tied embeddings, wrapper prefixes)
    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=str(src))
    set_model_state_dict(model, state["model"])
    enc = tiktoken.get_encoding("gpt2")
    items: Iterable[dict[str, Any]] = iter_items(data)
    if limit:
        items = (it for i, it in enumerate(items) if i < limit)

    def autocast():  # type: ignore[no-untyped-def]
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        )

    result = evaluate(model, items, enc.encode, device, autocast=autocast)
    extra = json.loads((src / "extra.json").read_text())
    record = {"event": "hellaswag", "checkpoint": src.name, "step": extra["step"],
              "tokens_seen": extra["tokens_seen"], **result}  # fmt: skip
    with (run / "log.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record
