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
