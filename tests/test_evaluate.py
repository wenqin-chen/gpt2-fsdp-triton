from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gptfsdp.data import shard_name, write_shard
from gptfsdp.evaluate import hf_to_ours, ours_to_hf, val_loss
from gptfsdp.model import GPT, GPTConfig

TINY = GPTConfig(block_size=16, vocab_size=97, n_layer=2, n_head=2, n_embd=32)


def test_hf_mapping_round_trip_reproduces_the_model() -> None:
    torch.manual_seed(0)
    ref = GPT(TINY)
    hf = ours_to_hf(ref.state_dict())
    assert hf["transformer.h.0.attn.c_attn.weight"].shape == (32, 96)  # Conv1D layout (in, out)
    assert "transformer.lm_head.weight" not in hf
    hf["transformer.h.0.attn.bias"] = torch.ones(1, 1, 16, 16)  # causal-mask buffer: ignored
    back = GPT(TINY)
    back.load_state_dict(hf_to_ours(hf, TINY.n_layer))
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    assert torch.allclose(ref(idx)[0], back(idx)[0])
    with pytest.raises(KeyError):
        hf_to_ours({"transformer.wte.weight": torch.zeros(3, 3)}, TINY.n_layer)


def test_val_loss_uses_the_same_windows_for_every_model(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    write_shard(tmp_path / shard_name("t", "val", 0), rng.integers(0, 97, size=200))
    write_shard(tmp_path / shard_name("t", "train", 1), rng.integers(0, 97, size=200))
    torch.manual_seed(0)
    model = GPT(TINY)
    out = val_loss(model, tmp_path, tokens=160, seq_len=16, batch=3)
    assert out["tokens"] == 10 * 16  # ten full windows
    assert out["val_loss"] == pytest.approx(math.log(97), abs=0.5)
    assert out["perplexity"] == pytest.approx(math.exp(out["val_loss"]))
    again = val_loss(model, tmp_path, tokens=160, seq_len=16, batch=10)
    assert again["val_loss"] == pytest.approx(out["val_loss"], rel=1e-6)  # batching-invariant


def test_arc_rows_become_variable_width_items() -> None:
    from gptfsdp.arc import to_item
    from gptfsdp.hellaswag import render

    row = {
        "id": "q1",
        "question": "Which is a gas?",
        "choices": {"text": ["ice", "steam", "rock"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    item = to_item(row)
    assert item["ctx"] == "Question: Which is a gas?\nAnswer:" and item["label"] == 1
    tokens, mask, label = render(item, lambda s: [len(w) % 16 for w in s.split()])
    assert tokens.shape[0] == 3 and mask.shape[0] == 3 and label == 1  # three choices, not four
    five = {**row, "choices": {"text": list("abcde"), "label": list("12345")}, "answerKey": "4"}
    assert to_item(five)["label"] == 3 and len(to_item(five)["endings"]) == 5
    with pytest.raises(ValueError, match="not among"):
        to_item({**row, "answerKey": "E"})


def test_load_run_model_restores_the_trained_weights(tmp_path: Path) -> None:
    from gptfsdp.data import synthetic_corpus
    from gptfsdp.evaluate import evaluate_checkpoint, load_run_model
    from gptfsdp.train import TrainConfig, train

    data = synthetic_corpus(tmp_path / "data", n_train_shards=1, shard_tokens=4096, vocab=256)
    cfg = TrainConfig(
        data_dir=str(data), out_dir=str(tmp_path / "runs"), run_name="r", preset="tiny",
        vocab_size=256, seq_len=32, micro_batch=4, total_batch_tokens=128, max_steps=8,
        warmup_steps=2, max_lr=3e-3, device="cpu", bf16=False, val_every=0, val_steps=2,
    )  # fmt: skip
    summary = train(cfg)
    model, extra = load_run_model(tmp_path / "runs" / "r", torch.device("cpu"))
    assert extra["step"] == 8 and extra["checkpoint"] == "step_0000008"
    held = evaluate_checkpoint(tmp_path / "runs" / "r", data, tokens=4 * 32 * 2)
    # the in-run validation used the same first tokens of the val shard (2 x micro-batch 4)
    assert held["val_loss"] == pytest.approx(summary["final_val_loss"], rel=1e-4)
    assert held["event"] == "heldout" and held["step"] == 8
    assert held["device"] == "cpu" and held["git_sha"] and held["time"]
