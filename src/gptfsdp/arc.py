"""ARC-Easy (AI2, CC BY-SA 4.0) as a zero-shot multiple-choice eval, same harness for every model.

Each question becomes the context ``"Question: {question}\\nAnswer:"`` and each choice the ending
``" {text}"`` (the lm-evaluation-harness prompt); the predicted choice has the lowest mean token
loss (``acc_norm`` here is token-length normalised, ``acc`` uses the summed loss). Items come from
the Hugging Face dataset ``allenai/ai2_arc`` (config ``ARC-Easy``), stored locally as parquet.
Scores from this harness are compared only with scores from this harness (this model and
OpenAI's GPT-2 124M), never with numbers produced by other evaluation code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

REPO_ID = "allenai/ai2_arc"
SPLIT_FILES = {"test": "ARC-Easy/test-00000-of-00001.parquet"}


def to_item(row: dict[str, Any]) -> dict[str, Any]:
    """One ARC row -> the ``render`` item format (ctx, endings, label)."""
    labels = list(row["choices"]["label"])
    if row["answerKey"] not in labels:
        raise ValueError(f"{row.get('id')}: answer {row['answerKey']!r} not among {labels}")
    return {
        "id": row.get("id"),
        "ctx": f"Question: {row['question']}\nAnswer:",
        "endings": list(row["choices"]["text"]),
        "label": labels.index(row["answerKey"]),
    }


def iter_items(path: str | Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for row in pq.read_table(path).to_pylist():
        yield to_item(row)


def download(out_dir: str | Path, split: str = "test") -> dict[str, Any]:
    """Fetch one ARC-Easy split (parquet) at a pinned revision; returns path, revision, sha256."""
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(REPO_ID).sha
    path = Path(
        hf_hub_download(
            REPO_ID, SPLIT_FILES[split], repo_type="dataset", revision=revision,
            local_dir=str(out_dir),
        )
    )  # fmt: skip
    return {
        "path": str(path),
        "revision": revision,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def evaluate_model(
    model: torch.nn.Module, data: str | Path, device: torch.device
) -> dict[str, float]:
    import tiktoken

    from gptfsdp.hellaswag import evaluate

    enc = tiktoken.get_encoding("gpt2")

    def autocast():  # type: ignore[no-untyped-def]
        return torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        )

    return evaluate(model.to(device), iter_items(data), enc.encode, device, autocast=autocast)


def evaluate_run_or_gpt2(
    data: str | Path, run: str | Path | None = None, gpt2: str | Path | None = None,
    out: str | Path = "baselines/openai_gpt2_arc_easy.json",
) -> dict[str, Any]:  # fmt: skip
    """ARC-Easy for a run's newest checkpoint (appended to its log) or for OpenAI's GPT-2 124M."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (run is None) == (gpt2 is None):
        raise ValueError("pass exactly one of run / gpt2")
    from gptfsdp.evaluate import provenance

    if run is not None:
        from gptfsdp.evaluate import load_run_model

        model, extra = load_run_model(run, device)
        record = {"event": "arc_easy", "checkpoint": extra["checkpoint"], "step": extra["step"]}
        record.update(evaluate_model(model, data, device))
        record.update(provenance(device))
        with (Path(run) / "log.jsonl").open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record
    from gptfsdp.evaluate import load_openai_gpt2

    assert gpt2 is not None
    record = {"event": "arc_easy", "model": "openai-community/gpt2 (GPT-2 124M, MIT)"}
    record.update(evaluate_model(load_openai_gpt2(gpt2), data, device))
    record.update(provenance(device))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
