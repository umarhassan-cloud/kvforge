"""Attention backends.

``gather``  — :func:`pulsecache.cache.paged_attention.paged_attention`: materializes
              the full gathered context (simple, fast for short contexts on CPU).
``online``  — :func:`paged_attention_online`: streams the block table in chunks
              with a numerically-stable online softmax (Milakov & Gimelshein 2018,
              the core of FlashAttention / flash-decoding). Peak extra memory is
              ``O(chunk)`` instead of ``O(context)``; this is the algorithm a fused
              CUDA/Triton kernel implements per thread block.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from pulsecache.cache.paged_attention import AttentionMetadata, pad_queries, paged_attention
from pulsecache.config import AttentionBackend

AttentionFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, AttentionMetadata, float], torch.Tensor
]


def paged_attention_online(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
    blocks_per_chunk: int = 4,
) -> torch.Tensor:
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads, block_size = key_cache.shape[2], key_cache.shape[1]
    group = num_heads // num_kv_heads
    b, max_blocks = meta.block_tables.shape
    device = query.device

    q, valid = pad_queries(query, meta)
    qlen = q.shape[1]
    q = q.view(b, qlen, num_kv_heads, group, head_dim).permute(0, 2, 3, 1, 4).float()

    q_idx = torch.arange(qlen, device=device)
    q_pos = (meta.context_lens - meta.query_lens)[:, None] + q_idx[None, :]  # [B, Q]

    m = torch.full((b, num_kv_heads, group, qlen, 1), float("-inf"), device=device)
    l_sum = torch.zeros_like(m)
    acc = torch.zeros((b, num_kv_heads, group, qlen, head_dim), device=device)

    for start in range(0, max_blocks, blocks_per_chunk):
        tables = meta.block_tables[:, start : start + blocks_per_chunk]
        nb = tables.shape[1]
        k = key_cache[tables].reshape(b, nb * block_size, num_kv_heads, head_dim)
        v = value_cache[tables].reshape(b, nb * block_size, num_kv_heads, head_dim)
        k, v = k.transpose(1, 2).float(), v.transpose(1, 2).float()

        k_pos = start * block_size + torch.arange(nb * block_size, device=device)
        mask = (k_pos[None, None, :] <= q_pos[:, :, None]) & (
            k_pos[None, None, :] < meta.context_lens[:, None, None]
        )
        s = torch.einsum("bhgqd,bhld->bhgql", q, k) * scale
        s = s.masked_fill(~mask[:, None, None], float("-inf"))

        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        # Rows with no visible key yet keep m == -inf; guard exp(-inf - -inf) = nan.
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        p = torch.exp(s - m_safe)
        correction = torch.exp(m - m_safe)
        l_sum = l_sum * correction + p.sum(dim=-1, keepdim=True)
        acc = acc * correction + torch.einsum("bhgql,bhld->bhgqd", p, v)
        m = m_new

    out = (acc / l_sum.clamp_min(1e-30)).to(query.dtype)
    out = out.permute(0, 3, 1, 2, 4).reshape(b, qlen, num_heads, head_dim)
    return out[valid]


def get_attention_backend(backend: AttentionBackend | str) -> AttentionFn:
    backend = AttentionBackend(backend)
    if backend == AttentionBackend.GATHER:
        return paged_attention
    if backend == AttentionBackend.ONLINE:
        return paged_attention_online
    raise ValueError(f"unknown attention backend {backend}")
