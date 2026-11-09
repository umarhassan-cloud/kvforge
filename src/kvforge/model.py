"""Llama/Gemma-style decoder with two forward paths.

* :meth:`Transformer.forward` — the serving path: flat varlen tokens, K/V written into
  a :class:`~kvforge.cache.PagedKVCache` and read back through PagedAttention.
* :meth:`Transformer.forward_dense` — HF-style padded ``[B, T]`` path with an optional
  contiguous ``past_key_values`` list; used by the naive baseline, perplexity eval and
  as a numerical oracle in tests.

All projection layers are ``nn.Linear`` so quantization can swap them in place.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from kvforge.cache.kv_cache import PagedKVCache
from kvforge.cache.paged_attention import AttentionMetadata
from kvforge.config import AttentionBackend, ModelConfig
from kvforge.kernels.attention import get_attention_backend

DenseKV = list[tuple[torch.Tensor, torch.Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, weight_offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) - weight_offset)
        self.eps = eps
        self.weight_offset = weight_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (self.weight.float() + self.weight_offset)).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.float()[..., None] * self.inv_freq.float()
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x: [..., T, H, D]``; ``cos/sin: [..., T, D]``."""
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend) -> None:
        super().__init__()
        assert cfg.head_dim is not None
        self.num_heads, self.num_kv_heads, self.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, h, bias=False)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_fn = get_attention_backend(backend)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        n = x.shape[0]
        q = apply_rope(self.q_proj(x).view(n, self.num_heads, self.head_dim), cos, sin)
        k = apply_rope(self.k_proj(x).view(n, self.num_kv_heads, self.head_dim), cos, sin)
        v = self.v_proj(x).view(n, self.num_kv_heads, self.head_dim)
        key_cache, value_cache = kv
        PagedKVCache.write(key_cache, value_cache, k, v, meta.slot_mapping)
        out = self.attn_fn(q, key_cache, value_cache, meta, self.scale)
        return self.o_proj(out.reshape(n, -1))

    def forward_dense(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        q = apply_rope(self.q_proj(x).view(b, t, self.num_heads, self.head_dim), cos, sin)
        k = apply_rope(self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim), cos, sin)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim)
        if past is not None:
            # HF "DynamicCache"-style growth: O(T) realloc + copy every step.
            k = torch.cat([past[0], k], dim=1)
            v = torch.cat([past[1], v], dim=1)
        group = self.num_heads // self.num_kv_heads
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2).repeat_interleave(group, dim=1)
        vh = v.transpose(1, 2).repeat_interleave(group, dim=1)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[:, None])
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        if cfg.hidden_act == "silu":
            self.act = F.silu
        elif cfg.hidden_act in ("gelu_tanh", "gelu_pytorch_tanh"):
            self.act = lambda t: F.gelu(t, approximate="tanh")
        else:
            raise ValueError(f"unsupported activation {cfg.hidden_act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, backend: AttentionBackend) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_weight_offset)
        self.self_attn = Attention(cfg, backend)
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_weight_offset
        )
        self.mlp = MLP(cfg)


class Transformer(nn.Module):
    def __init__(
        self, cfg: ModelConfig, backend: AttentionBackend = AttentionBackend.GATHER
    ) -> None:
        super().__init__()
        assert cfg.head_dim is not None
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, backend) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, cfg.norm_weight_offset)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def set_attention_backend(self, backend: AttentionBackend) -> None:
        for layer in self.layers:
            layer.self_attn.attn_fn = get_attention_backend(backend)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        if self.config.scale_embeddings:
            h = h * torch.tensor(math.sqrt(self.config.hidden_size), dtype=h.dtype)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: PagedKVCache,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        """Paged forward over flat tokens. Returns final hidden states ``[N, hidden]``."""
        h = self._embed(input_ids)
        cos, sin = self.rotary(positions)
        for i, layer in enumerate(self.layers):
            h = h + layer.self_attn(layer.input_layernorm(h), cos, sin, kv_cache.layer(i), meta)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return self.norm(h)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()

    def forward_dense(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DenseKV | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DenseKV]:
        """Padded dense forward.

        ``attention_mask`` is the HF-style ``[B, past + T]`` key-padding mask (1 = real token).
        Returns logits ``[B, T, V]`` and the grown contiguous cache.
        """
        b, t = input_ids.shape
        past_len = past_key_values[0][0].shape[1] if past_key_values else 0
        total = past_len + t
        if attention_mask is None:
            attention_mask = torch.ones(b, total, dtype=torch.long, device=input_ids.device)
        if positions is None:
            positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, past_len:]
        q_pos = torch.arange(past_len, total, device=input_ids.device)
        k_pos = torch.arange(total, device=input_ids.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        mask = causal[None] & attention_mask.bool()[:, None, :]
        # Fully-masked rows (left padding queries) would produce NaN; let them see themselves.
        mask = mask | (~mask.any(-1, keepdim=True) & (k_pos[None, None, :] == q_pos[None, :, None]))

        h = self._embed(input_ids)
        cos, sin = self.rotary(positions)
        new_past: DenseKV = []
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            a, kv = layer.self_attn.forward_dense(layer.input_layernorm(h), cos, sin, mask, past)
            h = h + a
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            new_past.append(kv)
        return self.compute_logits(self.norm(h)), new_past
