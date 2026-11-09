# kvforge design doc

| | |
|---|---|
| **Status** | v0.1, all core features implemented and tested on CPU |
| **Scope** | Single-node, single-GPU (or CPU) serving of decoder-only LMs in the 1–8B range |
| **Non-goals (v0.1)** | Tensor/pipeline parallelism, custom CUDA kernels, multi-LoRA, vision inputs |

## 1. Problem

Serving an LLM is a **memory-bandwidth and memory-capacity problem**, not a FLOPs problem:

* **Decode is bandwidth-bound.** Each generated token streams every weight byte plus that
  sequence's whole KV cache from HBM. At batch 1, an 8B fp16 model on an A100 uses <1% of its
  tensor-core FLOPs. Throughput only comes from batching many sequences per weight read.
* **Batch size is capped by KV memory.** Llama-3.1-8B needs 128 KiB of KV per token (fp16,
  GQA-8). A 4k-context sequence costs 512 MiB. Pre-reserving `max_model_len` per sequence (what
  a contiguous cache must do) wastes 60–90% of KV memory on real traffic, which directly
  caps batch size and therefore throughput.
* **Static batching wastes compute.** With `generate()`-style batching, a batch runs until its
  longest member finishes; short requests idle in padded slots and new requests queue.

kvforge implements the four techniques that production engines (vLLM, TGI, TensorRT-LLM, SGLang)
use to attack these, from scratch and in readable PyTorch, with tests that pin down correctness.

## 2. Requirements

| Requirement | How it is met | Verified by |
|---|---|---|
| Outputs identical to naive decoding (greedy) | Same model math in paged and dense paths | `test_batched_output_equals_isolated_greedy`, `test_paged_model_forward_matches_dense` |
| Bounded KV waste | Fixed-size blocks; waste ≤ `block_size - 1` tokens/seq | `test_fragmentation_metrics`, `scripts/profile_memory.py` |
| Requests join/leave every iteration | Iteration-level scheduler + varlen flat batch | `test_requests_join_running_batch` |
| Never deadlock / OOM under memory pressure | Recompute preemption of lowest-priority sequence; admission control | `test_preemption_preserves_outputs` |
| Speculation is lossless | Leviathan rejection sampling | `test_rejection_sampler_preserves_target_distribution` (200k-sample Monte-Carlo) |
| Quantization never worse than RTN on calibration data | AWQ grid search includes α=0 | `test_awq_scale_never_worse_than_rtn...` |

## 3. Architecture

See [docs/architecture.md](docs/architecture.md) for the module map and data flow. In short:

```
HTTP (serving/api.py) ─▶ AsyncEngine ─▶ InferenceEngine.step()
                                          ├─ Scheduler.schedule()        (what runs this iteration)
                                          ├─ PagedKVCache.copy_blocks()  (copy-on-write)
                                          ├─ ModelRunner.execute()       (one varlen forward)
                                          ├─ SpeculativeDecoder.step()   (k draft + 1 verify)
                                          └─ postprocess                 (stop checks, free blocks)
```

## 4. Key decisions and trade-offs

### 4.1 KV layout: `[num_blocks, block_size, kv_heads, head_dim]` per layer
* **Chosen:** one tensor pair per layer, flat *slot* addressing (`block * block_size + offset`).
  Writes are a single `index_copy_`; reads gather by block table.
* **Alternative:** one giant `[layers, 2, blocks, ...]` tensor (vLLM v0). Slightly better for
  swap-to-CPU, worse for readability; no throughput difference in PyTorch.
* **Block size 16 default.** Smaller blocks cut internal fragmentation (measured: 0.9% at 8, 1.8%
  at 16, 7.2% at 64) but lengthen block tables and make each gather less coalesced on GPU. 16 is
  also vLLM's default; the measured knee is between 16 and 32.

### 4.2 Flat varlen batches, no padding
Every forward receives `input_ids` as the concatenation of all scheduled tokens (prefill
chunks, decodes, speculative verifies) and an `AttentionMetadata` with `query_start_loc`,
`context_lens`, `block_tables`, `slot_mapping`. MLP/projection layers see one `[N, hidden]`
matrix (maximal GEMM efficiency); only attention needs per-sequence structure. This is what
makes *continuous* batching free: adding a request is appending rows, not reshaping a batch.

### 4.3 One attention entry point for all phases
Query `i` of a sequence sits at absolute position `context_len − query_len + i`, and attends to
cached keys `≤` that position. The same masked kernel therefore serves prefill, chunked prefill
on cached context, decode, and the `k+1`-token speculative verify. Fewer code paths, fewer bugs;
the cost is that pure-prefill batches do some redundant masked work vs. a dedicated
FlashAttention prefill kernel (listed in roadmap).

### 4.4 Scheduler: token budget, decode-first, recompute preemption
* **Token budget** (`max_num_batched_tokens`) bounds per-iteration latency; chunked prefill
  splits long prompts so one 8k prompt cannot stall every decoding stream (bounded TPOT p99).
* **Decode-first:** running sequences are scheduled before admitting new prompts. This protects
  inter-token latency of in-flight requests at a small TTFT cost to newcomers.
* **Preemption by recompute, not swap.** Freed blocks are simply dropped; on re-admission the
  prompt *and generated tokens* are prefilled again. For ≤8B models prefill runs at
  compute-bound speed, so recompute of a few hundred tokens is cheaper and far simpler than
  PCIe swap. Victim = lowest priority / latest arrival; never admit new work in a step that
  preempted (prevents thrash).
* **Head-of-line admission.** If the queue head does not fit, nothing behind it is admitted.
  This keeps FCFS/priority guarantees; the cost is some idle capacity with mixed prompt sizes.

### 4.5 Reference counting, copy-on-write, prefix caching
Blocks carry `ref_count`. `fork()` shares a sequence's blocks (parallel sampling / beam
search); the first write into a shared block triggers a CoW copy that the block manager
*reports* as `(src, dst)` and the cache executes — keeping the block manager tensor-free and
unit-testable in microseconds.

Automatic prefix caching keys **full, computed** blocks by a chained content hash
`h_i = hash(h_{i-1}, tokens_i)`, so a hit on block *i* implies the whole prefix matches.
Freed cached blocks go to an LRU evictable pool rather than the free list. Two invariants
keep it correct:
1. The block holding a prompt's last token is never reused (its logits must be recomputed and
   that write must not land in a shared block).
2. Any block about to be written loses its hash first.

Measured effect with a 512-token shared system prompt: concurrency in a fixed 16 GiB budget goes
from 137 to 318 sequences (`scripts/profile_memory.py`).

### 4.6 Speculative decoding: shared block tables, separate KV, no rollback
* Draft and target share **one BlockManager** (same block tables) but own separate KV
  tensors, so there is exactly one allocation decision per token.
* The scheduler reserves `k` lookahead slots per decoding sequence.
* Rejected drafts leave stale K/V beyond `num_computed_tokens`; it is overwritten by later
  writes, so no rollback copy is needed.
* The draft tracks its own `num_draft_computed_tokens`; when all `k` drafts are accepted the
  draft lags by one token and catches up in the next step's first draft forward (query_len 2).
* Default draft = **layer-skip self-draft** (first *n* layers + shared embeddings/LM head): zero
  extra weight memory, no second checkpoint. Any `Transformer` with the same vocab works.

### 4.7 Quantization: weight-only, module swap
Decode is bandwidth-bound, so shrinking *weights* is what buys speed; activation quantization
mostly helps compute-bound prefill and costs accuracy. INT8 is symmetric per-output-channel,
whose scale commutes out of the GEMM (`(x @ Qᵀ) * s`). INT4 is asymmetric group-wise (g=128,
packed two-per-byte, ≈0.56 B/weight). AWQ-style scale search protects salient input channels.

**Honest caveat:** on CPU (and in this PyTorch reference path on GPU) INT4 is
dequantize-then-GEMM, which saves memory but not time. The speed win requires a fused kernel
(Marlin, tinygemm/`torch._weight_int4pack_mm`) that unpacks in registers.

## 5. Correctness strategy

1. **Oracles.** Every optimized path is checked against a deliberately naive one: contiguous
   attention (`reference_attention`), dense full-recompute greedy decoding (`dense_greedy`).
2. **Exact equality, not tolerance, for end-to-end greedy outputs** across: batching, chunked
   prefill, preemption, prefix caching, both attention backends, speculation with k∈{1,3,5}.
3. **Statistical test for the sampler theorem.** 200k Monte-Carlo samples confirm the first
   emitted token is distributed as `p` and the acceptance rate equals `Σ min(p, q)`.
4. **Fragmented pools in tests.** Paged attention tests deliberately interleave junk
   allocations so block tables are non-contiguous.

## 6. Performance model and known gaps vs vLLM

On the same GPU we expect kvforge at roughly **60–70% of vLLM decode throughput** (estimate
from the roofline model in the benchmark report, **not measured on a GPU in this repo**).
The gap is kernels and host overhead, not algorithms:

| Gap | Impact | Fix |
|---|---|---|
| Attention gathers KV into a temporary tensor, then attends (2 HBM passes) | largest at long context | fused paged-attention kernel (Triton), reads blocks in place |
| Python scheduling + `torch.tensor` metadata build every step | 0.3–1 ms/step; dominates small models | CUDA graphs for decode, persistent pinned buffers, async scheduling |
| INT4 dequantizes into a temp fp16 matrix | INT4 slower than fp16 instead of ~2.5× faster | Marlin / tinygemm kernel |
| No FlashAttention prefill | prefill-heavy TTFT | SDPA/FA2 for pure-prefill batches |
| Rejection sampler round-trips to host (`.tolist()`) | ~50 µs/step | keep accept/reject on device |

## 7. Future work
See the roadmap in the [README](README.md#roadmap).
