"""GPT-2 (small by default) in plain PyTorch.

Pre-LayerNorm transformer with tied input/output embeddings and GPT-2 initialisation (residual
projections scaled by 1/sqrt(2 * n_layer)). Attention goes through
``F.scaled_dot_product_attention`` (FlashAttention backend on CUDA). ``GPTConfig.norm`` selects
``nn.LayerNorm`` ("torch") or the fused Triton kernel ("triton", CUDA only).

FLOP accounting follows nanoGPT's ``estimate_mfu``: ``6 * N + 12 * L * H * Q * T`` per token, with
``N`` the parameter count excluding the position embedding (PaLM appendix B).
"""

from __future__ import annotations

import inspect
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2's 50,257 padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    norm: str = "torch"  # "torch" | "triton"
    activation_checkpointing: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


GPT2_SMALL = GPTConfig()


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "torch":
        return nn.LayerNorm(cfg.n_embd)
    if cfg.norm == "triton":
        from gptfsdp.kernels.layernorm import TritonLayerNorm

        return TritonLayerNorm(cfg.n_embd)
    raise ValueError(f"norm must be 'torch' or 'triton', got {cfg.norm!r}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.RESIDUAL_SCALE = True  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg)
        self.mlp = MLP(cfg)
        self.activation_checkpointing = cfg.activation_checkpointing

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            out: torch.Tensor = checkpoint(self._forward, x, use_reentrant=False)
            return out
        return self._forward(x)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig = GPT2_SMALL) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_SCALE", False):
                std *= 1.0 / math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            # the loss is computed in fp32 whatever the compute dtype of the logits
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # -- accounting ------------------------------------------------------------------------------

    def num_params(self, non_embedding: bool = True) -> int:
        """Unique parameters (the tied lm_head counts once); ``non_embedding`` drops ``wpe``."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def flops_per_token(self) -> float:
        cfg = self.cfg
        head_dim = cfg.n_embd // cfg.n_head
        return 6.0 * self.num_params() + 12.0 * cfg.n_layer * cfg.n_head * head_dim * cfg.block_size

    # -- optimizer -------------------------------------------------------------------------------

    def configure_optimizer(
        self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        """AdamW with decay on >= 2-D tensors only (matmul weights and embeddings)."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=fused)
