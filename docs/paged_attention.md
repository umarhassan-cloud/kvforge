# Paged KV cache and PagedAttention

Reference: Kwon et al., *Efficient Memory Management for Large Language Model Serving with
PagedAttention*, SOSP 2023.

## Why

A contiguous KV cache must reserve space for the longest sequence a request *might* reach,
because the tensor cannot grow in place. On real traffic most sequences end far below
`max_model_len`, so most of that reservation is never used. The paper reports 60–80% of KV
memory wasted by fragmentation and over-reservation in pre-paging systems; our simulation
(below) agrees.

Paging fixes this the way an OS does: allocate fixed-size **blocks** on demand and map each
sequence's *logical* blocks to arbitrary *physical* blocks through a **block table**.

## Data structures (`cache/block_manager.py`)

```
logical view (seq "a", 11 tokens, block_size 4)

  pos:   0 1 2 3 | 4 5 6 7 | 8 9 10 _
  lblk:     0    |    1    |    2
                  block_table["a"] = [7, 2, 12]

physical pool (key_cache[l]: [num_blocks, 4, kv_heads, head_dim])

  blk 2:  a4 a5 a6 a7        ref=1
  blk 7:  a0 a1 a2 a3        ref=2   <- shared with seq "b" (prefix hit or fork)
  blk 12: a8 a9 a10 --       ref=1   <- 1 slot of internal fragmentation
```

* **Slot** of token `p`: `table[p // bs] * bs + p % bs`. K/V writes are a single
  `index_copy_` over the flattened `[num_blocks * bs, kv_heads, head_dim]` view.
* **Free list** is LIFO (recently freed blocks are reused first — warm in L2 on GPU).
* **ref_count** enables sharing. A write to a block with `ref_count > 1` triggers
  **copy-on-write**: allocate a new block, report `(src, dst)` to the caller, remap.
* **Prefix cache**: full, computed blocks are keyed by a chained hash
  `h_i = hash(h_{i-1}, tokens_i)`. Freed cached blocks enter an LRU *evictable* pool and can be
  revived by a later request with the same prefix; they are only evicted when the free list is
  empty.

Waste is bounded by `block_size - 1` slots per sequence, independent of `max_model_len`.

## Attention over non-contiguous blocks (`cache/paged_attention.py`)

Inputs are flat varlen queries `[N, H, D]` plus metadata. For each sequence the kernel:

1. pads queries to `[B, Q_max, H, D]` (only for attention — linears never see padding);
2. gathers KV: `key_cache[block_tables] -> [B, max_blocks * bs, KVH, D]`;
3. builds a mask where query `i` (absolute position `ctx − q_len + i`) sees key `j` iff
   `j ≤ pos_i` and `j < ctx`;
4. computes GQA attention by viewing queries as `[B, KVH, group, Q, D]` (no KV repeat);
5. un-pads back to `[N, H, D]`.

The same code handles prefill, chunked prefill over cached context, decode and the
`k + 1`-token speculative verify.

### Online-softmax backend (`kernels/attention.py`)

`paged_attention_online` walks the block table in chunks and keeps a running max `m`, running
denominator `l` and un-normalized accumulator `acc`:

```
m' = max(m, max_j s_j)
l' = l · e^{m − m'} + Σ_j e^{s_j − m'}
acc' = acc · e^{m − m'} + Σ_j e^{s_j − m'} v_j
out = acc / l
```

Peak temporary memory is `O(chunk)` rather than `O(context)`. This is the exact recurrence a
fused flash-decoding CUDA/Triton kernel runs per thread block; here it is the readable PyTorch
version and is tested to match the gather path to 1e-5 (including chunks with no visible keys,
where `m = −∞` must not produce NaN).

## Measured: memory efficiency

`python scripts/profile_memory.py` drives the real `BlockManager` with a long-tailed synthetic
chat workload (log-normal prompt/output lengths, mean final length 473 tokens) against a 16 GiB
KV budget for a Llama-3.1-8B-shaped model (128 KiB/token fp16):

| Strategy | Concurrent sequences | KV wasted |
|---|---:|---:|
| Contiguous, reserve `max_model_len` = 4096 | 32 | 88.5% |
| Contiguous, reserve prompt + `max_tokens` (1.5× mean) | 184 | 33.3% |
| **Paged, block 16** | **319.5** | **1.8%** |
| Paged, block 8 / 32 / 64 | 323.2 / 312.1 / 298.8 | 0.9% / 3.7% / 7.2% |
| Paged, bs 16 + shared 512-token system prompt, no prefix cache | 136.6 | — |
| **…with prefix caching** | **317.8** | — |

Concurrency is what bounds decode throughput (it sets the batch size), so a 10× concurrency
gain over max-length reservation is the headline systems win, before any kernel work.

## Correctness tests

* `test_paged_matches_contiguous_reference` — both backends × {decode, prefill, chunked,
  mixed} on a deliberately fragmented pool vs a contiguous oracle.
* `test_paged_model_forward_matches_dense` — full transformer, chunked prefill + decodes, logits
  vs HF-style dense forward.
* `test_fork_shares_blocks_and_copy_on_write`, `test_prefix_cache_*` — sharing semantics.
