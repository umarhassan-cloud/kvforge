"""Continuous-batching execution: turn a heterogeneous set of per-sequence work items
(prefill chunks, decodes, speculative verifies) into ONE flat varlen forward pass.

There is no padding along the token dimension: ``input_ids`` is the concatenation of
every item's tokens, and :class:`~kvforge.cache.AttentionMetadata` tells attention
where each sequence begins and which physical blocks hold its context. This is what
lets new requests join a running batch at any iteration without recompiling shapes
or padding everyone to the longest prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from kvforge.cache.block_manager import BlockManager
from kvforge.cache.kv_cache import PagedKVCache
from kvforge.cache.paged_attention import AttentionMetadata
from kvforge.model import Transformer

LogitsMode = Literal["last", "all", "none"]


@dataclass
class BatchItem:
    seq_id: str
    token_ids: list[int]
    start_pos: int  # absolute position of token_ids[0]
    logits: LogitsMode = "last"


@dataclass
class ModelInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    meta: AttentionMetadata
    logits_indices: torch.Tensor
    logits_counts: list[int]  # rows of the logits tensor that belong to each item


def prepare_inputs(
    items: Sequence[BatchItem], block_manager: BlockManager, device: torch.device | str
) -> ModelInput:
    input_ids: list[int] = []
    positions: list[int] = []
    slots: list[int] = []
    context_lens: list[int] = []
    starts = [0]
    tables: list[list[int]] = []
    logits_idx: list[int] = []
    counts: list[int] = []
    bs = block_manager.block_size

    for item in items:
        n = len(item.token_ids)
        end = item.start_pos + n
        offset = len(input_ids)
        input_ids.extend(item.token_ids)
        positions.extend(range(item.start_pos, end))
        slots.extend(block_manager.slots(item.seq_id, item.start_pos, end))
        context_lens.append(end)
        starts.append(offset + n)
        # Only the blocks covering the live context; lookahead blocks are not gathered.
        tables.append(block_manager.get_block_table(item.seq_id)[: -(-end // bs)])
        if item.logits == "last":
            logits_idx.append(offset + n - 1)
            counts.append(1)
        elif item.logits == "all":
            logits_idx.extend(range(offset, offset + n))
            counts.append(n)
        else:
            counts.append(0)

    width = max(len(t) for t in tables)
    padded = [t + [0] * (width - len(t)) for t in tables]
    as_long = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)  # noqa: E731
    meta = AttentionMetadata(
        slot_mapping=as_long(slots),
        block_tables=as_long(padded),
        context_lens=as_long(context_lens),
        query_start_loc=as_long(starts),
        max_query_len=max(len(i.token_ids) for i in items),
    )
    return ModelInput(as_long(input_ids), as_long(positions), meta, as_long(logits_idx), counts)


class ModelRunner:
    """Owns one model + its paged KV cache. Draft and target each get a runner; they
    share the *block manager* (same block tables) but not the KV tensors."""

    def __init__(
        self,
        model: Transformer,
        kv_cache: PagedKVCache,
        block_manager: BlockManager,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.device = device
        self.num_forward_passes = 0
        self.num_tokens_processed = 0

    @torch.inference_mode()
    def execute(self, items: Sequence[BatchItem]) -> torch.Tensor:
        """Run one forward; returns fp32 logits ``[sum(logits_counts), vocab]``."""
        inp = prepare_inputs(items, self.block_manager, self.device)
        hidden = self.model(inp.input_ids, inp.positions, self.kv_cache, inp.meta)
        self.num_forward_passes += 1
        self.num_tokens_processed += int(inp.input_ids.numel())
        if inp.logits_indices.numel() == 0:
            return hidden.new_empty((0, self.model.config.vocab_size), dtype=torch.float32)
        return self.model.compute_logits(hidden[inp.logits_indices])
