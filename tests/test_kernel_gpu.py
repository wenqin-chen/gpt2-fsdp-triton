"""Triton LayerNorm against PyTorch (CUDA only: ``pytest -m gpu`` on a GPU node)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("shape", [(7, 768), (1024, 768), (4096, 1000), (33, 4096)])
def test_forward_and_backward_match_pytorch(
    shape: tuple[int, int], dtype: torch.dtype, tol: float
) -> None:
    from gptfsdp.kernels.layernorm import layer_norm

    torch.manual_seed(0)
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=dtype, requires_grad=True)
    w = (1 + 0.1 * torch.randn(n, device="cuda", dtype=dtype)).requires_grad_()
    b = (0.1 * torch.randn(n, device="cuda", dtype=dtype)).requires_grad_()
    dy = torch.randn(m, n, device="cuda", dtype=dtype)
    y = layer_norm(x, w, b)
    y.backward(dy)
    xr, wr, br = (t.detach().float().requires_grad_() for t in (x, w, b))
    yr = F.layer_norm(xr, (n,), wr, br, 1e-5)
    yr.backward(dy.float())
    scale_w = max(1.0, wr.grad.abs().max().item())  # type: ignore[union-attr]
    assert (y.float() - yr).abs().max().item() < tol
    assert (x.grad.float() - xr.grad).abs().max().item() < tol * 4  # type: ignore[union-attr]
    assert (w.grad.float() - wr.grad).abs().max().item() < tol * scale_w * 4  # type: ignore[union-attr]
    assert (b.grad.float() - br.grad).abs().max().item() < tol * scale_w * 4  # type: ignore[union-attr]


@cuda
def test_model_with_triton_norm_matches_torch_norm() -> None:
    from gptfsdp.model import GPT, GPTConfig

    torch.manual_seed(0)
    cfg = GPTConfig(block_size=64, vocab_size=512, n_layer=2, n_head=4, n_embd=128)
    ref = GPT(cfg).cuda()
    tri = GPT(GPTConfig(**{**cfg.to_dict(), "norm": "triton"})).cuda()  # type: ignore[arg-type]
    tri.load_state_dict(ref.state_dict())
    idx = torch.randint(0, 512, (4, 64), device="cuda")
    _, l1 = ref(idx, idx)
    _, l2 = tri(idx, idx)
    l1.backward()  # type: ignore[union-attr]
    l2.backward()  # type: ignore[union-attr]
    assert torch.allclose(l1, l2, atol=1e-5)  # type: ignore[arg-type]
    for p1, p2 in zip(ref.parameters(), tri.parameters(), strict=True):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4)  # type: ignore[arg-type]
