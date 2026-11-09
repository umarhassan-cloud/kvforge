"""Physical KV-block allocator with per-sequence block tables.

This is the bookkeeping half of PagedAttention: KV memory is carved into
fixed-size *physical blocks*; every sequence owns a *block table* mapping its
logical block index ``pos // block_size`` to a physical block id. Blocks are
reference counted so that

* ``fork`` can share all blocks between a parent and child sequence
  (parallel sampling / beam search) with copy-on-write on first divergent write;
* automatic prefix caching can share *full* blocks between unrelated requests
  whose prompts start with the same tokens (content-hash keyed, LRU evicted).

The manager never touches tensors. Copy-on-write is reported to the caller as
``(src_block, dst_block)`` pairs which the KV cache executes before the forward.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass


class NoFreeBlocksError(RuntimeError):
    pass


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    # Content hash of (prefix hash, tokens in this block) once the block is full and computed.
    content_hash: int | None = None


@dataclass(frozen=True)
class BlockManagerStats:
    num_blocks: int
    block_size: int
    num_used_blocks: int
    num_free_blocks: int
    num_cached_free_blocks: int
    num_logical_blocks: int
    num_tokens: int
    utilization: float
    internal_fragmentation: float
    sharing_ratio: float
    prefix_cache_hit_rate: float

    @property
    def num_allocated_slots(self) -> int:
        return self.num_used_blocks * self.block_size


class BlockManager:
    """Reference-counted physical block pool + logical-to-physical block tables."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_caching: bool = False,
        watermark: float = 0.0,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.watermark_blocks = int(watermark * num_blocks)

        self._blocks = [PhysicalBlock(i) for i in range(num_blocks)]
        # LIFO free list: recently freed blocks are reused first (warm in L2 on GPU).
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        # ref_count == 0 but content still valid; reusable on a prefix hit, evicted LRU.
        self._evictable: OrderedDict[int, None] = OrderedDict()
        self._hash_to_block: dict[int, int] = {}

        self._tables: dict[str, list[int]] = {}
        self._num_tokens: dict[str, int] = {}
        # Chain hashes of each sequence's leading full computed blocks.
        self._seq_hashes: dict[str, list[int]] = {}

        self._prefix_queries = 0
        self._prefix_hits = 0

    # ------------------------------------------------------------------ queries
    @property
    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._evictable)

    def has_seq(self, seq_id: str) -> bool:
        return seq_id in self._tables

    def get_block_table(self, seq_id: str) -> list[int]:
        return self._tables[seq_id]

    def ref_count(self, block_id: int) -> int:
        return self._blocks[block_id].ref_count

    def slot(self, seq_id: str, position: int) -> int:
        """Flat slot index (``block_id * block_size + offset``) for a token position."""
        table = self._tables[seq_id]
        return table[position // self.block_size] * self.block_size + position % self.block_size

    def slots(self, seq_id: str, start: int, end: int) -> list[int]:
        table, bs = self._tables[seq_id], self.block_size
        return [table[p // bs] * bs + p % bs for p in range(start, end)]

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    # --------------------------------------------------------------- allocation
    def _pop_free_block(self) -> int:
        if self._free:
            block_id = self._free.pop()
        elif self._evictable:
            block_id, _ = self._evictable.popitem(last=False)
            block = self._blocks[block_id]
            if block.content_hash is not None:
                self._hash_to_block.pop(block.content_hash, None)
                block.content_hash = None
        else:
            raise NoFreeBlocksError("KV cache exhausted")
        self._blocks[block_id].ref_count = 1
        return block_id

    def _release(self, block_id: int) -> None:
        block = self._blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"double free of block {block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.content_hash is not None and self.enable_prefix_caching:
                self._evictable[block_id] = None
            else:
                block.content_hash = None
                self._free.append(block_id)

    def _block_hashes(self, token_ids: Sequence[int], num_full_blocks: int) -> list[int]:
        hashes, prev = [], 0
        bs = self.block_size
        for i in range(num_full_blocks):
            prev = hash((prev, tuple(token_ids[i * bs : (i + 1) * bs])))
            hashes.append(prev)
        return hashes

    def _cacheable_blocks(self, num_tokens: int) -> int:
        # Never reuse the block holding the last token: the model must recompute
        # at least one token to produce logits, and that write must not hit a shared block.
        return max(0, (num_tokens - 1) // self.block_size)

    def _prefix_hits_for(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        if not self.enable_prefix_caching:
            return [], []
        hits, hashes = [], []
        for h in self._block_hashes(token_ids, self._cacheable_blocks(len(token_ids))):
            block_id = self._hash_to_block.get(h)
            if block_id is None:
                break
            hits.append(block_id)
            hashes.append(h)
        return hits, hashes

    def can_allocate(self, token_ids: Sequence[int]) -> bool:
        hits, _ = self._prefix_hits_for(token_ids)
        # Hits on in-use blocks are free; hits on evictable blocks still consume pool capacity.
        active_hits = sum(1 for b in hits if self._blocks[b].ref_count > 0)
        needed = self.blocks_needed(len(token_ids)) - active_hits
        return needed <= self.num_free_blocks - self.watermark_blocks

    def allocate(self, seq_id: str, token_ids: Sequence[int]) -> int:
        """Create a block table for ``token_ids``; returns number of prefix-cached tokens."""
        if seq_id in self._tables:
            raise ValueError(f"sequence {seq_id} already allocated")
        hits, hit_hashes = self._prefix_hits_for(token_ids)
        if self.enable_prefix_caching:
            self._prefix_queries += self._cacheable_blocks(len(token_ids))
            self._prefix_hits += len(hits)
        total = self.blocks_needed(len(token_ids))
        active_hits = sum(1 for b in hits if self._blocks[b].ref_count > 0)
        if total - active_hits > self.num_free_blocks:
            raise NoFreeBlocksError(f"cannot allocate {total} blocks for {seq_id}")

        table: list[int] = []
        for block_id in hits:
            block = self._blocks[block_id]
            if block.ref_count == 0:
                self._evictable.pop(block_id, None)
            block.ref_count += 1
            table.append(block_id)
        for _ in range(total - len(hits)):
            table.append(self._pop_free_block())
        self._tables[seq_id] = table
        self._num_tokens[seq_id] = len(token_ids)
        self._seq_hashes[seq_id] = hit_hashes
        return len(hits) * self.block_size

    def _cow_needed(self, table: list[int], write_start: int, total_tokens: int) -> int:
        first = write_start // self.block_size
        last = min(len(table), self.blocks_needed(total_tokens))
        return sum(1 for i in range(first, last) if self._blocks[table[i]].ref_count > 1)

    def can_append_slots(self, seq_id: str, total_tokens: int, write_start: int) -> bool:
        table = self._tables[seq_id]
        new_blocks = max(0, self.blocks_needed(total_tokens) - len(table))
        return new_blocks + self._cow_needed(table, write_start, total_tokens) <= self.num_free_blocks

    def append_slots(
        self, seq_id: str, total_tokens: int, write_start: int
    ) -> list[tuple[int, int]]:
        """Ensure capacity for ``total_tokens`` and exclusive ownership of blocks written
        from ``write_start`` on. Returns copy-on-write ``(src, dst)`` block pairs."""
        if not self.can_append_slots(seq_id, total_tokens, write_start):
            raise NoFreeBlocksError(f"cannot append slots for {seq_id}")
        table = self._tables[seq_id]
        copies: list[tuple[int, int]] = []
        first = write_start // self.block_size
        hashes = self._seq_hashes.get(seq_id)
        if hashes is not None and len(hashes) > first:
            del hashes[first:]
        for i in range(first, min(len(table), self.blocks_needed(total_tokens))):
            block = self._blocks[table[i]]
            if block.ref_count > 1:
                new_id = self._pop_free_block()
                copies.append((table[i], new_id))
                block.ref_count -= 1
                table[i] = new_id
            elif block.content_hash is not None:
                # About to be overwritten: its cached content is no longer trustworthy.
                self._hash_to_block.pop(block.content_hash, None)
                block.content_hash = None
        while len(table) < self.blocks_needed(total_tokens):
            table.append(self._pop_free_block())
        self._num_tokens[seq_id] = max(self._num_tokens[seq_id], total_tokens)
        return copies

    def fork(self, parent_id: str, child_id: str) -> None:
        """Share every block of ``parent_id`` with a new sequence ``child_id``."""
        if child_id in self._tables:
            raise ValueError(f"sequence {child_id} already allocated")
        table = list(self._tables[parent_id])
        for block_id in table:
            self._blocks[block_id].ref_count += 1
        self._tables[child_id] = table
        self._num_tokens[child_id] = self._num_tokens[parent_id]
        self._seq_hashes[child_id] = list(self._seq_hashes.get(parent_id, []))

    def mark_computed(self, seq_id: str, token_ids: Sequence[int], num_computed: int) -> None:
        """Record live token count and, with prefix caching, register content hashes of
        full computed blocks so later requests can reuse them. Incremental: O(new blocks)."""
        table = self._tables[seq_id]
        self._num_tokens[seq_id] = min(len(token_ids), len(table) * self.block_size)
        if not self.enable_prefix_caching:
            return
        hashes = self._seq_hashes.setdefault(seq_id, [])
        num_full = min(num_computed // self.block_size, len(table))
        bs = self.block_size
        for i in range(len(hashes), num_full):
            prev = hashes[-1] if hashes else 0
            h = hash((prev, tuple(token_ids[i * bs : (i + 1) * bs])))
            hashes.append(h)
            block = self._blocks[table[i]]
            if block.content_hash is None and h not in self._hash_to_block:
                block.content_hash = h
                self._hash_to_block[h] = table[i]

    def free(self, seq_id: str) -> None:
        table = self._tables.pop(seq_id, None)
        self._num_tokens.pop(seq_id, None)
        self._seq_hashes.pop(seq_id, None)
        if table is None:
            return
        # Free tail first so that, under LRU eviction, prefix blocks outlive suffix blocks.
        for block_id in reversed(table):
            self._release(block_id)

    def reset_prefix_cache(self) -> None:
        for block_id in list(self._evictable):
            self._blocks[block_id].content_hash = None
            self._free.append(block_id)
        self._evictable.clear()
        self._hash_to_block = {
            h: b for h, b in self._hash_to_block.items() if self._blocks[b].ref_count > 0
        }

    # ------------------------------------------------------------------ metrics
    def stats(self) -> BlockManagerStats:
        used = self.num_blocks - self.num_free_blocks
        logical = sum(len(t) for t in self._tables.values())
        # Count each physical block's filled slots once (shared blocks are shared tokens).
        filled: dict[int, int] = {}
        for seq_id, table in self._tables.items():
            n = self._num_tokens[seq_id]
            for i, block_id in enumerate(table):
                in_block = min(self.block_size, max(0, n - i * self.block_size))
                filled[block_id] = max(filled.get(block_id, 0), in_block)
        tokens = sum(filled.values())
        slots = used * self.block_size
        return BlockManagerStats(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            num_used_blocks=used,
            num_free_blocks=self.num_free_blocks,
            num_cached_free_blocks=len(self._evictable),
            num_logical_blocks=logical,
            num_tokens=tokens,
            utilization=used / self.num_blocks,
            internal_fragmentation=(1.0 - tokens / slots) if slots else 0.0,
            sharing_ratio=(logical / used) if used else 1.0,
            prefix_cache_hit_rate=(self._prefix_hits / self._prefix_queries)
            if self._prefix_queries
            else 0.0,
        )
