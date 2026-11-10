# pulsecache benchmark report

> **Read this first: what is measured and what is not.**
>
> | Section | Source | Hardware |
> |---|---|---|
> | §2 Throughput / latency, §3 speculative sweep, §4 quantization accuracy | **Measured** with `benchmarks/run_bench.py` | Laptop CPU, 1 thread (no GPU available) |
> | §5 KV memory efficiency | **Measured** by driving the real `BlockManager` with a synthetic workload (`scripts/profile_memory.py`) | Model-free, shape math for Llama-3.1-8B |
> | §6 GPU throughput and cost | **Analytic roofline projection — NOT measured** | Assumed L4 / A100 / H100 specs |
>
> All model weights are **random** (no checkpoint downloads), shaped as described below. Random
> weights preserve the cost profile of each technique (FLOPs, bytes moved, scheduling
> behaviour) but not trained-model *accuracy* characteristics; §1.3 explains the two
> adjustments made so acceptance-rate and perplexity measurements are still meaningful.

## 1. Methodology

### 1.1 Environment

```
python 3.10.11, torch 2.13.0+cpu, Intel64 Family 6 Model 140 (laptop), torch.set_num_threads(1)
```

Single-threaded because on this machine OpenMP fan-out costs more than it saves at these
matrix sizes (measured: 16-request decode was 1.6× faster with 1 thread than 4). The
`--threads` flag is recorded in the result JSON.

### 1.2 Model and workload

* **Model:** `small` preset: Llama-architecture decoder, 26.1M params, hidden 512, 6 layers,
  8 query heads / 4 KV heads (GQA), SwiGLU 1408, vocab 8192, fp32.
* **Workload:** 48 requests, all arriving at t=0 (offline burst), prompt lengths uniform in
  [16, 128], output lengths uniform in [8, 128], `ignore_eos`, greedy decoding. The same seeded
  workload is used for every mode.
* **Engine settings:** `max_num_seqs=16`, `max_num_batched_tokens=1024`, block size 16,
  1024 blocks, chunked prefill on, FCFS.
* **Baseline (`naive`):** `benchmarks/baselines.py`. It reproduces HF `generate()` serving:
  static batches of 16 in arrival order, left padding, finished rows keep computing,
  `torch.cat`-grown contiguous KV. It uses **the same weights** through
  `Transformer.forward_dense`, so the comparison isolates the serving strategy.
* **Metrics:** output tokens/s over wall-clock; TTFT (arrival→first token, includes queueing);
  TPOT (mean inter-token time per request); E2E; peak KV bytes actually allocated;
  $/1M output tokens = `hourly_price / (tok/s · 3600) · 1e6` at $0.40/h (16-vCPU on-demand).
* **Warm-up:** one short request per engine before timing; counters reset afterwards.

### 1.3 Making random weights a fair stand-in

1. **Sharpened LM head (×8).** A random model's next-token distribution is nearly uniform
   (entropy ≈ ln 8192 ≈ 9 nats), which makes perplexity uninformative and every draft token a
   coin flip over 8k options. Scaling the LM head gives ~3.8 nats, which is in the range of
   trained LMs on web text.
2. **Damped upper layers (`--residual-scale`).** In trained LLMs, early-layer predictions
   correlate strongly with final ones; this is why layer-skip/early-exit drafting works.
   Random layers have no such correlation. We multiply the residual writes (`o_proj`,
   `down_proj`) of layers above the draft depth by a scale *s*, which gives a single dial
   from "draft is useless" to "draft is nearly perfect". Default *s* = 0.05 gives
   **α ≈ 0.6–0.67** per-token acceptance, within the 0.6–0.8 range reported for
   Llama-family draft pairs. §3 sweeps *s* and reports α explicitly.

Neither adjustment changes the FLOPs or bytes of any mode. Both apply identically to all modes.

### 1.4 Reproduce

```bash
python benchmarks/run_bench.py --threads 1 --num-requests 48 --prompt-len 16 128 \
    --output-len 8 128 --sweep-k 2 4 --sweep-requests 4 --tag small-cpu
python scripts/profile_memory.py            # §5
```

Raw results: [`results/small-cpu.json`](results/small-cpu.json). **Variance:** this laptop is
noisy; across two full runs, absolute throughput moved by up to ±12% (e.g. `continuous` 84.6 vs
94.9 tok/s). Ratios between modes were stable to about ±0.1×. Treat single-digit-percent
differences as noise.

## 2. Throughput and latency (measured, CPU)

| mode | out tok/s | vs naive | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | E2E p99 ms | mean batch | peak KV MiB | weights MiB | $/1M tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| naive (HF-style static) | 59.6 | 1.00× | 19,246 | 36,210 | 227.8 | 50,763 | 16.0 | 47.1 | 99.5 | 1.86 |
| **pulsecache continuous** | **84.6** | **1.42×** | **9,340** | 27,401 | **157.6** | **35,654** | 13.0 | **24.9** | 99.5 | **1.31** |
| pulsecache + speculative (k=4) | 78.5 | 1.32× | 10,402 | 27,998 | 171.1 | 38,480 | 12.6 | 25.5 | 99.5 | 1.41 |
| pulsecache + INT8 | 66.4 | 1.11× | 12,590 | 34,770 | 206.7 | 45,336 | 13.0 | 24.9 | 49.1 | 1.67 |
| pulsecache + INT4 (g=128) | 40.2 | 0.68× | 19,441 | 53,811 | 336.5 | 74,580 | 13.0 | 24.9 | 41.2 | 2.76 |

What the numbers show:

* **Continuous batching: 1.42× throughput, 2.1× lower median TTFT, 1.9× less peak KV.** The
  static baseline spends **45% of its token computations** on left-padding or on rows that
  have already finished, and a new batch cannot start until the slowest request in the
  current one finishes. All modes generate the same 3,026 tokens. pulsecache refills
  a free slot on the next iteration. Its mean batch (13.0) is below the cap (16) only because
  the burst drains at the end.
* **Speculative decoding at batch 16 is roughly neutral on CPU (0.93× of continuous).** With
  α = 0.67 each verify emits 2.61 tokens, but on a CPU the 5-token verify for 16 sequences is
  compute-bound and costs close to 5× a 1-token decode. The premise that "k + 1 tokens cost
  about the same as 1" only holds in the bandwidth-bound regime. §3 shows the win at batch
  size 1, which is where speculation is meant to be used.
* **Weight-only quantization saves memory, not time, on this path.** Projection weights shrink
  3.9× (INT8) and 7.3× (INT4) once you subtract the 32 MiB of unquantized fp32 embedding and
  LM head. The INT8 forward casts int8→fp32 per call; the INT4 forward unpacks and
  dequantizes per call. Both add work on a compute-bound CPU. This matches the analytic model
  in §6: speed gains need fused dequant-GEMM kernels.

## 3. Speculative decoding sweep (measured, CPU, batch size 1)

Layer-skip self-draft (1 of 6 layers, shared embeddings and LM head; draft/target cost ratio
c ≈ 0.33). 4 requests × 64 tokens per cell. "theory" is Leviathan et al.'s
`(1 − α^{k+1}) / (1 − α)` evaluated at the measured per-token α.

| residual scale | k | accepted / drafted | α (per-token) | tokens/step measured | tokens/step theory | TPOT base ms | TPOT spec ms | E2E speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 2 | 0.16 | 0.26 | 1.32 | 1.33 | 39.3 | 53.4 | 0.86× |
| 0.15 | 4 | 0.08 | 0.25 | 1.33 | 1.33 | 39.3 | 77.1 | 0.56× |
| 0.07 | 2 | 0.47 | 0.59 | 1.94 | 1.94 | 35.9 | 36.7 | 0.98× |
| 0.07 | 4 | 0.29 | 0.58 | 2.18 | 2.21 | 35.9 | 49.6 | 0.70× |
| 0.05 | 2 | 0.49 | 0.60 | 1.98 | 1.97 | 39.9 | 38.9 | 1.12× |
| 0.05 | 4 | 0.35 | 0.63 | 2.38 | 2.42 | 39.9 | 46.1 | 0.95× |
| **0.03** | **2** | 0.68 | **0.77** | **2.36** | 2.36 | 39.7 | **33.4** | **1.26×** |
| 0.03 | 4 | 0.54 | 0.76 | 3.16 | 3.12 | 39.7 | 34.7 | 1.20× |

* **Measured tokens/step matches theory to within 2% in every cell.** This is a strong
  end-to-end check that the rejection sampler, lookahead slot reservation and KV bookkeeping
  are correct. A bug that dropped accepted tokens or corrupted the draft cache would show up
  as a gap here.
* Note the gap between *accepted/drafted* and *α*. Drafting stops at the first rejection, so
  the naive ratio understates α (0.54 vs 0.76 at k=4). The speedup formula needs α. Reporting
  the naive ratio as "acceptance rate" is a common mistake.
* Break-even is α ≈ 0.55–0.6 at this draft cost. Larger k only pays when α is high, because
  each extra draft costs 0.33 of a target step. This is why production systems tune k per
  workload. Adaptive k is on the roadmap.
* On a GPU, verify cost is nearly flat in k at small batch, and a draft with c ≈ 0.05–0.1
  (e.g. 1B drafting for 8B) is typical. The same α = 0.77, k = 4 would then give roughly
  3.1 / (1 + 4 · 0.08) ≈ 2.4× (projected, not measured).

## 4. Quantization accuracy (measured, CPU)

Held-out text = 16 sequences × 128 tokens **sampled from the fp32 model at T=0.8**. A random
model has no natural-language test set, so the honest held-out text is its own samples.
Calibration uses 8 other sampled sequences. Same model as §2 (default damping).

| mode | weights MiB | PPL | ΔPPL | top-1 agreement | KL(fp32 ‖ quant) |
|---|---:|---:|---:|---:|---:|
| fp32 | 99.5 | 34.81 | — | 1.000 | 0 |
| INT8 per-channel, RTN | 49.0 | 34.91 | +0.3% | 0.964 | 0.0017 |
| INT8 per-channel, AWQ | 49.1 | 34.92 | +0.3% | 0.963 | 0.0017 |
| INT4 g=128, RTN | 41.1 | 46.60 | +34% | 0.563 | 0.305 |
| INT4 g=128, AWQ | 41.2 | 46.31 | +33% | 0.579 | 0.303 |
| INT4 g=32, RTN | 43.1 | 43.03 | +24% | 0.640 | 0.197 |
| INT4 g=32, AWQ | 43.2 | 42.98 | +23% | 0.637 | 0.204 |

* **INT8 is effectively lossless** (+0.3% PPL), consistent with the literature for weight-only
  per-channel INT8.
* **INT4 here is much worse than published results for real models**, where ΔPPL is typically
  around 2–5% at g=128 with AWQ/GPTQ. This is expected, and it is a property of the stand-in
  model, not of the quantizer:
  - i.i.d. Gaussian weights spread information evenly across all weights, so 4-bit rounding
    noise (≈10% relative per weight at g=128) lands everywhere. Trained weights are
    heavy-tailed and low-rank, so most of the signal survives;
  - the ×8 sharpened LM head amplifies hidden-state noise into logit noise;
  - AWQ helps little because random activations have no salient outlier channels, which are
    the thing AWQ protects. `tests/test_quant.py::test_awq_scale_never_worse_than_rtn...`
    plants 4 outlier channels and AWQ then cuts the output MSE by more than 10%.
* **The group-size trade-off is real and visible:** g=32 costs 5% more memory than g=128 and
  removes about a third of the KL (0.305 → 0.197).

## 5. KV-cache memory efficiency (measured with the real BlockManager)

`scripts/profile_memory.py`: Llama-3.1-8B KV shape (128 KiB/token fp16), 16 GiB KV budget,
1,500 requests with log-normal prompt/output lengths (mean final length 473 tokens,
`max_model_len` 4096). Concurrency is measured while the queue is saturated, and includes
recompute preemption.

| Strategy | Concurrent sequences | KV wasted |
|---|---:|---:|
| Contiguous, reserve `max_model_len` | 32 | 88.5% |
| Contiguous, reserve prompt + `max_tokens` (1.5 × mean) | 184 | 33.3% |
| Paged, block 8 | 323.2 | 0.9% |
| **Paged, block 16 (default)** | **319.5** | **1.8%** |
| Paged, block 32 | 312.1 | 3.7% |
| Paged, block 64 | 298.8 | 7.2% |
| Paged bs16, shared 512-token system prompt, no prefix cache | 136.6 | — |
| **…with prefix caching** | **317.8** | — |

**10× more concurrent sequences than max-length contiguous reservation, and 1.7× more than
exact-size reservation.** Decode throughput scales almost linearly with batch size until the
GPU becomes compute-bound, so this is the single largest throughput lever in the engine.
Prefix caching recovers 2.3× concurrency when requests share a system prompt.

## 6. GPU projection (ANALYTIC ROOFLINE — NOT MEASURED)

No GPU was available. Below is a bandwidth roofline for **Llama-3.1-8B, batch 32, context
1024, fp16 KV**, produced by `run_bench.roofline()`:

```
tok/s = batch · efficiency · HBM_BW / (weight_bytes + batch · ctx · 128 KiB)
```

**Assumptions:**

* *Efficiency* = 0.80 for vLLM (fused paged-attention and GEMM kernels, CUDA graphs) and 0.52
  for pulsecache (two-pass gather attention, Python step overhead).
* *Weight bytes per parameter:* fused kernels read the packed format once (2 / 1 / 0.56 B for
  fp16 / INT8 / INT4). pulsecache's reference path reads the packed weights, writes an fp16
  temporary and reads it back (2 / 5 / 4.56 B).
* Prices: L4 $0.80/h, A100-80G $3.67/h, H100 $6.98/h (on-demand list prices).

| GPU | weights | vLLM tok/s | pulsecache tok/s | pulsecache / vLLM | vLLM $/1M | pulsecache $/1M |
|---|---|---:|---:|---:|---:|---:|
| L4 | fp16 | 377 | 245 | 65% | 0.59 | 0.91 |
| A100-80G | fp16 | 2,515 | 1,635 | 65% | 0.41 | 0.62 |
| H100 | fp16 | 4,213 | 2,739 | 65% | 0.46 | 0.71 |
| A100-80G | INT8 | 4,154 | 749 | 18% | 0.25 | 1.36 |
| A100-80G | INT4 | 5,824 | 813 | 14% | 0.18 | 1.25 |

**Target: 60–70% of vLLM fp16 decode throughput on the same GPU.** The 65% figure is an
*input* of this model (the efficiency ratio), chosen from where the known gaps below put us;
it is not a measurement. The quantized rows show the model's most important lesson: without
a fused kernel, weight-only quantization **reduces** throughput (≈0.5× of fp16), even though it
quarters weight memory. This is the same direction §2 measured on CPU.

### Gap analysis vs vLLM, in priority order

| # | Gap | Why it costs | Fix | Expected gain |
|---|---|---|---|---|
| 1 | INT8/INT4 dequantize to an fp16 temporary | 2.5× more weight traffic than fp16 | Marlin / `torch._weight_int4pack_mm` / Triton fused dequant-GEMM | INT4 → ~2.2× fp16 decode |
| 2 | Gather-then-attend PagedAttention | KV read, written to a temporary, read again; padded to the longest context in the batch | Triton paged-attention kernel that reads blocks in place (the online-softmax recurrence is already implemented in `kernels/attention.py`) | 1.2–1.5× at ctx ≥ 1k |
| 3 | Python scheduling and metadata building every step | 0.3–1 ms/step, comparable to a whole decode step for 1B models | CUDA graphs for decode buckets, persistent pinned metadata buffers, overlap scheduling with the GPU | 1.1–1.3× small models |
| 4 | No FlashAttention for pure-prefill batches | masked dense prefill is O(T²) memory per sequence | dispatch pure prefills to SDPA / FA2 varlen | TTFT at long prompts |
| 5 | Sampler and rejection sampler sync to host | `.tolist()` per step | keep accept/reject on device, async detokenize | ~50 µs/step |

## 7. Threats to validity

* **CPU ≠ GPU.** CPU decode is compute-bound at these sizes; GPU decode is bandwidth-bound.
  Batching gains (§2, §5) carry over, and in fact grow on GPU. Quantization and speculation
  gains depend on the regime and are only projected for GPU (§6).
* **Random weights.** Mitigated by §1.3, and α is reported explicitly so that results can be
  re-read for any real draft pair. Quantization *accuracy* on random weights is pessimistic (§4).
* **Burst arrivals.** TTFT includes queueing behind a 48-request burst, so absolute TTFT numbers
  reflect the queue, not prefill speed. The ratios between modes are the meaningful part. A
  Poisson-arrival online benchmark is on the roadmap.
* **Small sample per sweep cell** (4 × 64 tokens). Tokens/step is very stable (it agrees with
  theory to 2%); TPOT has about ±10% noise.
