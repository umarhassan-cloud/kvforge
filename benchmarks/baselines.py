"""Baselines that pulsecache is measured against.

``naive_static_generate`` reproduces the HF ``model.generate()`` serving pattern:

* **static batching** — requests are grouped in arrival order into fixed batches; a batch
  runs until its *longest* request finishes and nobody new can join;
* **padding** — prompts are left-padded to the batch max, finished rows keep computing;
* **contiguous, growing KV** — ``torch.cat`` on every step (``DynamicCache`` semantics),
  so peak KV = ``batch * (max_prompt + max_new)`` regardless of actual lengths.

It reuses the exact same :class:`pulsecache.model.Transformer` weights through
``forward_dense`` so every difference in the numbers comes from the serving strategy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from pulsecache.model import Transformer
from pulsecache.sampling import SamplingParams, logits_to_probs, sample_from_probs


@dataclass
class BaselineRequest:
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float = 0.0
    output_token_ids: list[int] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None


@dataclass
class BaselineResult:
    requests: list[BaselineRequest]
    elapsed_s: float
    peak_kv_bytes: int
    wasted_token_steps: int  # row-steps spent on already-finished or padded sequences
    total_token_steps: int


@torch.inference_mode()
def naive_static_generate(
    model: Transformer,
    requests: list[BaselineRequest],
    batch_size: int,
    pad_token_id: int = 0,
    seed: int = 0,
) -> BaselineResult:
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    cfg = model.config
    assert cfg.head_dim is not None
    elem = next(model.parameters()).element_size()
    kv_bytes_per_token = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * elem

    start = time.perf_counter()
    for r in requests:
        r.arrival_time = start
    peak_kv = wasted = total = 0

    for b0 in range(0, len(requests), batch_size):
        batch = requests[b0 : b0 + batch_size]
        bsz = len(batch)
        max_prompt = max(len(r.prompt_token_ids) for r in batch)
        max_new = max(r.params.max_tokens for r in batch)
        ids = torch.full((bsz, max_prompt), pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros((bsz, max_prompt), dtype=torch.long, device=device)
        for i, r in enumerate(batch):
            n = len(r.prompt_token_ids)
            ids[i, max_prompt - n :] = torch.tensor(r.prompt_token_ids, device=device)
            mask[i, max_prompt - n :] = 1

        params = [r.params for r in batch]
        logits, past = model.forward_dense(ids, mask)
        done = torch.zeros(bsz, dtype=torch.bool, device=device)
        total += bsz * max_prompt
        wasted += int((mask == 0).sum())
        for step in range(max_new):
            probs = logits_to_probs(logits[:, -1], params)
            nxt = probs.argmax(-1) if all(p.greedy for p in params) else sample_from_probs(probs, gen)
            now = time.perf_counter()
            for i, r in enumerate(batch):
                if done[i]:
                    continue
                tok = int(nxt[i])
                r.output_token_ids.append(tok)
                if r.first_token_time is None:
                    r.first_token_time = now
                if len(r.output_token_ids) >= r.params.max_tokens:
                    done[i] = True
            if bool(done.all()) or step == max_new - 1:
                break
            nxt = torch.where(done, torch.full_like(nxt, pad_token_id), nxt)
            total += bsz
            wasted += int(done.sum())
            mask = torch.cat([mask, torch.ones((bsz, 1), dtype=mask.dtype, device=device)], dim=1)
            logits, past = model.forward_dense(nxt[:, None], mask, past)
        peak_kv = max(peak_kv, past[0][0].shape[1] * bsz * kv_bytes_per_token)
        # generate() returns the whole batch at once.
        end = time.perf_counter()
        for r in batch:
            r.finish_time = end
    return BaselineResult(requests, time.perf_counter() - start, peak_kv, wasted, total)
