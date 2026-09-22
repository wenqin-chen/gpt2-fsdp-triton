from __future__ import annotations

import math

import pytest
import torch

from gptfsdp.model import GPT, GPT2_SMALL, GPTConfig

TINY = GPTConfig(block_size=32, vocab_size=128, n_layer=2, n_head=2, n_embd=32)


def test_gpt2_small_parameter_count_and_tying() -> None:
    model = GPT(GPT2_SMALL)
    # GPT-2 124M has 124,439,808 parameters at vocab 50,257; padding to 50,304 adds 47 x 768
    assert model.num_params(non_embedding=False) == 124_439_808 + 47 * 768
    assert model.lm_head.weight is model.wte.weight
    head_dim = 768 // 12
    expected = 6 * model.num_params() + 12 * 12 * 12 * head_dim * 1024
    assert model.flops_per_token() == pytest.approx(expected)


def test_initial_loss_is_near_uniform_and_shapes() -> None:
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    targets = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))  # independent of the inputs
    logits, loss = model(idx, targets)
    assert logits.shape == (4, TINY.block_size, TINY.vocab_size)
    assert loss is not None and abs(loss.item() - math.log(TINY.vocab_size)) < 0.3
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.zeros((1, TINY.block_size + 1), dtype=torch.long))


def test_residual_projections_use_scaled_init() -> None:
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=8, n_head=4, n_embd=256, vocab_size=128, block_size=16))
    fc_std = model.h[0].mlp.c_fc.weight.std().item()
    proj_std = model.h[0].mlp.c_proj.weight.std().item()
    assert fc_std == pytest.approx(0.02, rel=0.05)
    assert proj_std == pytest.approx(0.02 / math.sqrt(2 * 8), rel=0.05)


def test_activation_checkpointing_matches_plain_forward_and_backward() -> None:
    torch.manual_seed(0)
    plain = GPT(TINY)
    ckpt = GPT(GPTConfig(**{**TINY.to_dict(), "activation_checkpointing": True}))  # type: ignore[arg-type]
    ckpt.load_state_dict(plain.state_dict())
    idx = torch.randint(0, TINY.vocab_size, (2, TINY.block_size))
    _, l1 = plain(idx, idx)
    _, l2 = ckpt(idx, idx)
    assert l1 is not None and l2 is not None
    l1.backward()
    l2.backward()
    assert torch.allclose(l1, l2)
    for p1, p2 in zip(plain.parameters(), ckpt.parameters(), strict=True):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6)  # type: ignore[arg-type]


def test_triton_norm_falls_back_on_cpu_with_identical_outputs() -> None:
    torch.manual_seed(0)
    ref = GPT(TINY)
    tri = GPT(GPTConfig(**{**TINY.to_dict(), "norm": "triton"}))  # type: ignore[arg-type]
    tri.load_state_dict(ref.state_dict())  # same parameter names as nn.LayerNorm
    idx = torch.randint(0, TINY.vocab_size, (2, TINY.block_size))
    assert torch.allclose(ref(idx)[0], tri(idx)[0], atol=1e-6)
    with pytest.raises(ValueError, match="norm must be"):
        GPT(GPTConfig(**{**TINY.to_dict(), "norm": "rms"}))  # type: ignore[arg-type]


def test_optimizer_groups_decay_only_matrices() -> None:
    model = GPT(TINY)
    opt = model.configure_optimizer(0.1, 6e-4, (0.9, 0.95), "cpu")
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    n = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n == model.num_params(non_embedding=False)
