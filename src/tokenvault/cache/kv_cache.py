"""Physical KV storage: one ``[num_blocks, block_size, num_kv_heads, head_dim]`` tensor
pair per layer, addressed through flat *slot* indices produced by the block manager."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.key_caches = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.value_caches = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

    @property
    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.key_caches + self.value_caches)

    @property
    def bytes_per_block(self) -> int:
        return self.memory_bytes // self.num_blocks

    def layer(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[idx], self.value_caches[idx]

    @staticmethod
    def write(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Scatter ``key/value`` of shape ``[N, kv_heads, head_dim]`` into flat slots."""
        nb, bs, h, d = key_cache.shape
        key_cache.view(nb * bs, h, d).index_copy_(0, slot_mapping, key.to(key_cache.dtype))
        value_cache.view(nb * bs, h, d).index_copy_(0, slot_mapping, value.to(value_cache.dtype))

    def copy_blocks(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Execute copy-on-write block copies ``src -> dst`` across all layers."""
        if not pairs:
            return
        device = self.key_caches[0].device
        src = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        dst = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        for k, v in zip(self.key_caches, self.value_caches):
            k[dst] = k[src]
            v[dst] = v[src]
