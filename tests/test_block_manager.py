import pytest

from tokenvault.cache.block_manager import BlockManager, NoFreeBlocksError


def test_allocate_and_free_roundtrip():
    bm = BlockManager(num_blocks=8, block_size=4)
    assert bm.allocate("a", list(range(10))) == 0
    assert len(bm.get_block_table("a")) == 3
    assert bm.num_free_blocks == 5
    bm.free("a")
    assert bm.num_free_blocks == 8


def test_slot_mapping_is_block_major():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate("a", list(range(6)))
    table = bm.get_block_table("a")
    assert bm.slot("a", 0) == table[0] * 4
    assert bm.slot("a", 5) == table[1] * 4 + 1
    assert bm.slots("a", 3, 6) == [table[0] * 4 + 3, table[1] * 4, table[1] * 4 + 1]


def test_append_slots_grows_only_at_block_boundary():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate("a", [1, 2, 3])
    assert bm.append_slots("a", 4, write_start=3) == []
    assert len(bm.get_block_table("a")) == 1
    bm.append_slots("a", 5, write_start=4)
    assert len(bm.get_block_table("a")) == 2


def test_exhaustion_is_detected():
    bm = BlockManager(num_blocks=2, block_size=4)
    bm.allocate("a", list(range(8)))
    assert not bm.can_allocate([1])
    assert not bm.can_append_slots("a", 9, write_start=8)
    with pytest.raises(NoFreeBlocksError):
        bm.allocate("b", [1])


def test_fork_shares_blocks_and_copy_on_write():
    bm = BlockManager(num_blocks=8, block_size=4)
    bm.allocate("parent", list(range(6)))  # blocks: [full, half]
    bm.fork("parent", "child")
    p_table = list(bm.get_block_table("parent"))
    assert bm.get_block_table("child") == p_table
    assert all(bm.ref_count(b) == 2 for b in p_table)

    # Child writes position 6 (inside the shared half block) -> CoW that block only.
    copies = bm.append_slots("child", 7, write_start=6)
    assert len(copies) == 1
    src, dst = copies[0]
    assert src == p_table[1]
    c_table = bm.get_block_table("child")
    assert c_table[0] == p_table[0] and c_table[1] == dst
    assert bm.ref_count(p_table[0]) == 2 and bm.ref_count(p_table[1]) == 1

    # Parent can now write in place; no copy needed.
    assert bm.append_slots("parent", 7, write_start=6) == []
    stats = bm.stats()
    assert stats.num_logical_blocks == 4 and stats.num_used_blocks == 3
    assert stats.sharing_ratio == pytest.approx(4 / 3)

    bm.free("parent")
    bm.free("child")
    assert bm.num_free_blocks == 8


def test_double_free_is_safe_noop_and_refcount_guarded():
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate("a", [1, 2])
    bm.free("a")
    bm.free("a")  # unknown sequence -> no-op
    assert bm.num_free_blocks == 4


def test_prefix_caching_reuses_full_blocks():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    prompt = list(range(100, 111))  # 11 tokens: 2 full blocks + 3
    assert bm.allocate("a", prompt) == 0
    bm.mark_computed("a", prompt, len(prompt))

    # Same 8-token prefix, different tail: both full blocks are reused.
    other = prompt[:8] + [7, 7, 7, 7]
    assert bm.allocate("b", other) == 8
    assert bm.get_block_table("b")[:2] == bm.get_block_table("a")[:2]
    assert bm.ref_count(bm.get_block_table("a")[0]) == 2
    assert bm.stats().prefix_cache_hit_rate > 0


def test_prefix_cache_never_shares_block_holding_last_token():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    prompt = list(range(8))  # exactly 2 full blocks
    bm.allocate("a", prompt)
    bm.mark_computed("a", prompt, 8)
    # Identical prompt: last block must be recomputed to produce logits -> only 1 block hit.
    assert bm.allocate("b", prompt) == 4


def test_prefix_cache_survives_free_and_is_lru_evicted():
    bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_caching=True)
    prompt = list(range(9))
    bm.allocate("a", prompt)
    bm.mark_computed("a", prompt, 9)
    bm.free("a")
    stats = bm.stats()
    assert stats.num_cached_free_blocks == 2  # two full blocks kept warm
    assert bm.num_free_blocks == 4

    assert bm.allocate("b", prompt) == 8  # revived from the evictable pool
    bm.free("b")
    # Fill the whole pool with unrelated content -> cached blocks are evicted.
    bm.allocate("c", list(range(500, 516)))
    bm.free("c")
    assert bm.allocate("d", prompt) == 0


def test_fragmentation_metrics():
    bm = BlockManager(num_blocks=16, block_size=8)
    bm.allocate("a", list(range(9)))  # 2 blocks, 9/16 slots used
    s = bm.stats()
    assert s.num_used_blocks == 2
    assert s.internal_fragmentation == pytest.approx(1 - 9 / 16)
    assert s.utilization == pytest.approx(2 / 16)
