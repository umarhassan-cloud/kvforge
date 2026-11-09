"""PagedAttention over non-contiguous KV blocks.

A single varlen entry point handles every phase the engine produces:

* prefill (query_len == context_len),
* chunked prefill on top of cached context (query_len < context_len),
* decode (query_len == 1),
* speculative verification (query_len == k + 1).

Queries arrive *flattened* (``[num_tokens, heads, head_dim]``, sequences
concatenated, delimited by ``query_start_loc``); the K/V of the current step
must already be written into the cache. Query ``i`` of a sequence sits at
absolute position ``context_len - query_len + i`` and attends causally to
every cached key at or before that position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor  # [num_tokens] flat slot each new token's K/V is written to
    block_tables: torch.Tensor  # [batch, max_blocks] physical block ids, padded with 0
    context_lens: torch.Tensor  # [batch] tokens in cache *after* this step's writes
    query_start_loc: torch.Tensor  # [batch + 1] prefix sum of query lengths
    max_query_len: int

    @property
    def batch_size(self) -> int:
        return int(self.block_tables.shape[0])

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


def pad_queries(query: torch.Tensor, meta: AttentionMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    """``[N, H, D]`` flat queries -> ``[B, Q, H, D]`` padded queries + ``[B, Q]`` validity mask."""
    q_idx = torch.arange(meta.max_query_len, device=query.device)
    valid = q_idx[None, :] < meta.query_lens[:, None]
    tok = torch.where(valid, meta.query_start_loc[:-1, None] + q_idx[None, :], 0)
    return query[tok], valid


def causal_context_mask(meta: AttentionMetadata, key_len: int, device: torch.device) -> torch.Tensor:
    """``[B, Q, L]`` boolean mask: key position ``j`` visible to padded query ``q``."""
    q_idx = torch.arange(meta.max_query_len, device=device)
    q_pos = (meta.context_lens - meta.query_lens)[:, None] + q_idx[None, :]
    k_pos = torch.arange(key_len, device=device)
    return (k_pos[None, None, :] <= q_pos[:, :, None]) & (
        k_pos[None, None, :] < meta.context_lens[:, None, None]
    )


def gather_kv(cache: torch.Tensor, block_tables: torch.Tensor) -> torch.Tensor:
    """Gather ``[num_blocks, bs, H, D]`` into per-sequence ``[B, max_blocks * bs, H, D]``."""
    b, nb = block_tables.shape
    _, bs, h, d = cache.shape
    return cache[block_tables].reshape(b, nb * bs, h, d)


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    meta: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    """Gather-then-attend PagedAttention with GQA. Returns ``[N, H, D]``."""
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = key_cache.shape[2]
    group = num_heads // num_kv_heads
    b = meta.batch_size

    q, valid = pad_queries(query, meta)  # [B, Q, H, D]
    qlen = q.shape[1]
    k = gather_kv(key_cache, meta.block_tables).transpose(1, 2)  # [B, KVH, L, D]
    v = gather_kv(value_cache, meta.block_tables).transpose(1, 2)
    q = q.view(b, qlen, num_kv_heads, group, head_dim).permute(0, 2, 3, 1, 4)  # [B,KVH,G,Q,D]

    scores = torch.einsum("bhgqd,bhld->bhgql", q.float(), k.float()) * scale
    mask = causal_context_mask(meta, k.shape[2], query.device)
    scores = scores.masked_fill(~mask[:, None, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhgql,bhld->bhgqd", probs, v.float()).to(query.dtype)
    out = out.permute(0, 3, 1, 2, 4).reshape(b, qlen, num_heads, head_dim)
    return out[valid]


def reference_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float, q_offset: int = 0
) -> torch.Tensor:
    """Contiguous single-sequence causal attention used as a correctness oracle.

    ``query: [Q, H, D]``, ``key/value: [L, KVH, D]``; query ``i`` is at position ``q_offset + i``.
    """
    group = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(group, dim=1)
    v = value.repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,lhd->hql", query.float(), k.float()) * scale
    q_pos = torch.arange(query.shape[0])[:, None] + q_offset
    k_pos = torch.arange(key.shape[0])[None, :]
    scores = scores.masked_fill(~(k_pos <= q_pos)[None], float("-inf"))
    return torch.einsum("hql,lhd->qhd", torch.softmax(scores, -1), v.float()).to(query.dtype)
