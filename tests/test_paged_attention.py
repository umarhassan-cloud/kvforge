import random

import pytest
import torch

from pulsecache.cache.block_manager import BlockManager
from pulsecache.cache.kv_cache import PagedKVCache
from pulsecache.cache.paged_attention import AttentionMetadata, paged_attention, reference_attention
from pulsecache.kernels.attention import paged_attention_online

H, KVH, D, BS = 4, 2, 8, 4


def _setup(seq_lens: list[int], query_lens: list[int], seed: int = 0):
    """Write random K/V for each sequence into *shuffled* physical blocks and build metadata."""
    g = torch.Generator().manual_seed(seed)
    bm = BlockManager(num_blocks=64, block_size=BS)
    # Fragment the pool so tables are non-contiguous.
    bm.allocate("junk", list(range(37)))
    rng = random.Random(seed)
    cache = PagedKVCache(1, 64, BS, KVH, D)
    keys, values, tables, slots_all = [], [], [], []
    for i, n in enumerate(seq_lens):
        bm.allocate(f"s{i}", list(range(n)))
        if rng.random() < 0.5:
            bm.free("junk")
            bm.allocate("junk", list(range(rng.randint(1, 20))))
        k = torch.randn(n, KVH, D, generator=g)
        v = torch.randn(n, KVH, D, generator=g)
        slots = torch.tensor(bm.slots(f"s{i}", 0, n))
        PagedKVCache.write(cache.key_caches[0], cache.value_caches[0], k, v, slots)
        keys.append(k)
        values.append(v)
        tables.append(bm.get_block_table(f"s{i}"))
        slots_all.append(slots)
    width = max(len(t) for t in tables)
    starts = [0]
    for q in query_lens:
        starts.append(starts[-1] + q)
    meta = AttentionMetadata(
        slot_mapping=torch.cat([s[-q:] for s, q in zip(slots_all, query_lens)]),
        block_tables=torch.tensor([t + [0] * (width - len(t)) for t in tables]),
        context_lens=torch.tensor(seq_lens),
        query_start_loc=torch.tensor(starts),
        max_query_len=max(query_lens),
    )
    query = torch.randn(sum(query_lens), H, D, generator=g)
    return cache, keys, values, meta, query


def _reference(keys, values, query, seq_lens, query_lens, scale):
    outs, off = [], 0
    for k, v, n, q in zip(keys, values, seq_lens, query_lens):
        outs.append(reference_attention(query[off : off + q], k, v, scale, q_offset=n - q))
        off += q
    return torch.cat(outs)


@pytest.mark.parametrize(
    "seq_lens,query_lens",
    [
        ([7, 13, 1, 20], [1, 1, 1, 1]),  # pure decode
        ([9, 16], [9, 16]),  # pure prefill
        ([12, 30, 5], [4, 7, 5]),  # chunked prefill / spec verify on top of cached context
        ([17, 3, 25], [1, 3, 5]),  # mixed batch
    ],
)
@pytest.mark.parametrize("backend", [paged_attention, paged_attention_online])
def test_paged_matches_contiguous_reference(seq_lens, query_lens, backend):
    cache, keys, values, meta, query = _setup(seq_lens, query_lens)
    scale = D**-0.5
    got = backend(query, cache.key_caches[0], cache.value_caches[0], meta, scale)
    want = _reference(keys, values, query, seq_lens, query_lens, scale)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_online_softmax_handles_chunk_with_no_visible_keys():
    # blocks_per_chunk=1 forces chunks that are fully masked for short sequences.
    cache, keys, values, meta, query = _setup([2, 23], [1, 3])
    got = paged_attention_online(query, cache.key_caches[0], cache.value_caches[0], meta, 0.3, 1)
    want = _reference(keys, values, query, [2, 23], [1, 3], 0.3)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(got).all()


def test_copy_blocks():
    cache = PagedKVCache(2, 4, BS, KVH, D)
    cache.key_caches[1][0].fill_(3.0)
    cache.copy_blocks([(0, 2)])
    assert torch.equal(cache.key_caches[1][2], cache.key_caches[1][0])


def test_paged_model_forward_matches_dense(tiny_model):
    """Full transformer: paged prefill + decode logits == dense recomputation logits."""
    from pulsecache.batching.continuous import BatchItem, ModelRunner

    m = tiny_model
    cfg = m.config
    bm = BlockManager(64, BS)
    cache = PagedKVCache(cfg.num_layers, 64, BS, cfg.num_kv_heads, cfg.head_dim)
    runner = ModelRunner(m, cache, bm)
    prompt = [5, 9, 2, 77, 31, 8, 100, 3, 64]
    bm.allocate("a", prompt + [0, 0])
    # Chunked prefill (5 + 4), then two single-token decodes.
    runner.execute([BatchItem("a", prompt[:5], 0, "none")])
    paged = [runner.execute([BatchItem("a", prompt[5:], 5, "all")])]
    paged.append(runner.execute([BatchItem("a", [11], 9, "last")]))
    paged.append(runner.execute([BatchItem("a", [12], 10, "last")]))
    got = torch.cat(paged)

    dense, _ = m.forward_dense(torch.tensor([prompt + [11, 12]]))
    want = dense[0, 5:11]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
