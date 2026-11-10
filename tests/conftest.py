from __future__ import annotations

import pytest
import torch

from pulsecache import PRESETS, CacheConfig, EngineConfig, SchedulerConfig, build_model

torch.set_num_threads(1)  # tiny ops: thread fan-out costs more than it saves


@pytest.fixture
def tiny_model():
    return build_model(PRESETS["tiny"], seed=0)


def make_config(model, **overrides) -> EngineConfig:
    cache = overrides.pop("cache", CacheConfig(block_size=4, num_blocks=256))
    sched = overrides.pop("scheduler", SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=256))
    return EngineConfig(model=model.config, cache=cache, scheduler=sched, **overrides)


@torch.inference_mode()
def dense_greedy(model, prompt: list[int], n: int) -> list[int]:
    """Reference: greedy decoding by full recomputation, no KV cache at all."""
    ids = torch.tensor([prompt])
    out = []
    for _ in range(n):
        logits, _ = model.forward_dense(ids)
        tok = int(logits[0, -1].argmax())
        out.append(tok)
        ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    return out
