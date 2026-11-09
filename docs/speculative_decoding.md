# Speculative decoding

References: Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative
Decoding*, ICML 2023; Chen et al., *Accelerating LLM Decoding with Speculative Sampling*, 2023.

## Idea

Decode is bandwidth-bound: scoring `k + 1` tokens in one forward costs about the same HBM
traffic as scoring 1. So let a cheap **draft** model guess `k` tokens, and have the **target**
score all of them at once. Accept the longest correct prefix, plus one token from the target.
With the right acceptance rule the output distribution is *exactly* the target's.

## Algorithm (`speculative/draft_verify.py`)

For each decoding sequence with last token `x` at position `n − 1`:

1. **Draft** `k` tokens autoregressively: `d_i ~ q_i`, keep the processed distributions `q_i`
   (temperature/top-k/top-p applied — p and q must be processed identically).
2. **Verify**: one target forward on `[x, d_1 … d_k]` at positions `n − 1 … n + k − 1`
   (query_len `k + 1` through the same PagedAttention kernel) → `p_1 … p_{k+1}`.
3. **Accept/reject**, left to right:
   * accept `d_i` with probability `min(1, p_i(d_i) / q_i(d_i))`;
   * at the first rejection, sample from the residual `norm(max(0, p_i − q_i))` and stop;
   * if all `k` are accepted, sample a bonus token from `p_{k+1}`.

Every step emits between 1 and `k + 1` tokens. The implementation is fully vectorized over the
batch: acceptance via `cumprod` over the accept mask, residual/bonus sampled with one gather.

**Greedy is a special case**, not a separate code path: greedy rows produce one-hot `p` and `q`,
so the ratio is 1 (accept) iff the draft's argmax equals the target's, and the residual is the
target's argmax.

### Why it is lossless (sketch)

`P(emit x) = q(x)·min(1, p(x)/q(x)) + (1 − Σ_y min(p(y), q(y))) · residual(x)`
`= min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x)`.
`test_rejection_sampler_preserves_target_distribution` checks this with 200k samples and also
checks the acceptance rate equals `Σ_x min(p(x), q(x))`.

## Expected speedup

With per-token acceptance rate `α` and `k` drafts, expected tokens per target forward:

```
E[tokens/step] = (1 − α^{k+1}) / (1 − α)
```

Wall-clock speedup = `E[tokens/step] / (1 + k · c)`, where `c` is the draft/target cost ratio.
The benchmark sweep reports measured tokens/step next to this formula.

## KV-cache integration

| Concern | Design |
|---|---|
| Memory for draft tokens | Scheduler reserves `k` lookahead slots per decode (`num_lookahead_slots`) |
| Draft KV | Separate tensors, **shared block tables** (one BlockManager, one allocation decision) |
| Rejected tokens | Target K/V beyond `num_computed_tokens` is stale and simply overwritten later — no rollback copy |
| Draft lag | `num_draft_computed_tokens = min(n + k − 1, n + accepted)`; after a full accept the draft is 1 behind and catches up with a 2-token query on the next step |
| Prefix cache | Only blocks computed by *both* models are registered (`min(target, draft)`) |
| Context limit | `k` is clamped per step to `max_model_len − num_tokens` |

## Draft model choices

* **Layer-skip self-draft (default):** the first *n* layers of the target plus its embeddings,
  final norm and LM head (`make_layer_skip_draft`). Zero extra weights, no second checkpoint.
  Works well when early-exit predictions correlate with final ones (true for trained LLMs, and
  improved further by fine-tuning an early-exit head — see LayerSkip, Elhoushi et al., 2024).
* **Separate small model:** pass any `Transformer` with the same vocabulary as `draft_model`
  (e.g. Llama-3.2-1B drafting for Llama-3.1-8B).

## When it helps — and when it doesn't

* Helps most at **small batch** (latency-bound, GPU underutilized) and high acceptance (code,
  structured output, low temperature).
* At **large batch** the target is already closer to compute-bound; verifying `k + 1` tokens
  per sequence is no longer free, and spending batch slots on likely-rejected tokens can reduce
  throughput. Production systems disable or shrink `k` as batch size grows; kvforge exposes `k`
  per engine and clamps it per step, adaptive `k` is on the roadmap.
* On **CPU** the verify forward is compute-bound, so the "k+1 tokens for the price of 1"
  premise only partially holds; the benchmark report quantifies this.

## Tests

* `test_greedy_speculative_output_equals_plain_greedy[k=1,3,5]` — bit-identical to dense greedy.
* `test_perfect_draft_accepts_all_and_cuts_target_passes` — draft == target ⇒ acceptance 1.0 and
  exactly `1 + ceil((N − 1)/(k + 1))` target forwards.
* `test_rejection_sampler_greedy_one_hot`, `..._identical_models_accept_everything`.
