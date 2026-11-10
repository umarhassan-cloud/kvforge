import random

import pytest
import torch

from tokenvault import (
    AttentionBackend,
    CacheConfig,
    InferenceEngine,
    RequestStatus,
    SamplingParams,
    SchedulerConfig,
)
from tokenvault.batching.continuous import BatchItem, prepare_inputs
from tokenvault.cache.block_manager import BlockManager

from .conftest import dense_greedy, make_config

GREEDY = SamplingParams(max_tokens=10, temperature=0.0)


def _prompts(n: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1, 512) for _ in range(rng.randint(2, 30))] for _ in range(n)]


def test_prepare_inputs_flattens_varlen_batch():
    bm = BlockManager(16, 4)
    bm.allocate("a", list(range(6)))
    bm.allocate("b", list(range(2)))
    inp = prepare_inputs(
        [BatchItem("a", [1, 2, 3], 3, "last"), BatchItem("b", [9], 1, "all")], bm, "cpu"
    )
    assert inp.input_ids.tolist() == [1, 2, 3, 9]
    assert inp.positions.tolist() == [3, 4, 5, 1]
    assert inp.meta.query_start_loc.tolist() == [0, 3, 4]
    assert inp.meta.context_lens.tolist() == [6, 2]
    assert inp.logits_indices.tolist() == [2, 3]
    assert inp.meta.slot_mapping.tolist() == bm.slots("a", 3, 6) + bm.slots("b", 1, 2)


@pytest.mark.parametrize("backend", [AttentionBackend.GATHER, AttentionBackend.ONLINE])
def test_batched_output_equals_isolated_greedy(tiny_model, backend):
    prompts = _prompts(6)
    want = [dense_greedy(tiny_model, p, 10) for p in prompts]
    engine = InferenceEngine(make_config(tiny_model, attention_backend=backend), model=tiny_model)
    outs = engine.generate(prompts, GREEDY)
    assert [o.output_token_ids for o in outs] == want


def test_requests_join_running_batch(tiny_model):
    prompts = _prompts(4, seed=1)
    want = [dense_greedy(tiny_model, p, 10) for p in prompts]
    engine = InferenceEngine(make_config(tiny_model), model=tiny_model)
    engine.add_request(prompts[0], GREEDY, request_id="r0")
    engine.add_request(prompts[1], GREEDY, request_id="r1")
    done = {}
    step = 0
    while engine.has_unfinished_requests():
        if step == 3:  # r0 / r1 are mid-decode: inject new work
            engine.add_request(prompts[2], GREEDY, request_id="r2")
        if step == 5:
            engine.add_request(prompts[3], GREEDY, request_id="r3")
        if step == 4:
            assert engine.scheduler.num_running == 3
        for out in engine.step():
            if out.finished:
                done[out.request_id] = out.output_token_ids
        step += 1
    assert [done[f"r{i}"] for i in range(4)] == want
    assert max(engine.metrics.batch_sizes) >= 3


def test_short_requests_leave_early_and_free_blocks(tiny_model):
    engine = InferenceEngine(make_config(tiny_model), model=tiny_model)
    engine.add_request([1, 2, 3], SamplingParams(max_tokens=2, temperature=0), request_id="short")
    engine.add_request([4, 5, 6], SamplingParams(max_tokens=12, temperature=0), request_id="long")
    finished_at = {}
    step = 0
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                finished_at[o.request_id] = step
        step += 1
    assert finished_at["short"] < finished_at["long"]
    bm = engine.block_manager
    assert bm.num_free_blocks == bm.num_blocks


def test_preemption_preserves_outputs(tiny_model):
    """KV pool far too small for all requests at once -> recompute preemption kicks in,
    but every request must still produce exactly the isolated greedy output."""
    prompts = _prompts(6, seed=2)
    want = [dense_greedy(tiny_model, p, 10) for p in prompts]
    cfg = make_config(
        tiny_model,
        cache=CacheConfig(block_size=4, num_blocks=14),
        scheduler=SchedulerConfig(max_num_seqs=6, max_num_batched_tokens=64),
    )
    engine = InferenceEngine(cfg, model=tiny_model)
    outs = engine.generate(prompts, GREEDY)
    assert [o.output_token_ids for o in outs] == want
    assert engine.scheduler.num_preemptions > 0


def test_chunked_prefill_preserves_outputs(tiny_model):
    prompts = [list(range(1, 60)), list(range(100, 140))]
    want = [dense_greedy(tiny_model, p, 10) for p in prompts]
    cfg = make_config(tiny_model, scheduler=SchedulerConfig(max_num_batched_tokens=16))
    engine = InferenceEngine(cfg, model=tiny_model)
    assert [o.output_token_ids for o in engine.generate(prompts, GREEDY)] == want


def test_prefix_caching_preserves_outputs_and_hits(tiny_model):
    system = list(range(200, 232))  # 32-token shared system prompt = 8 blocks
    prompts = [system + [i, i + 1, i + 2] for i in range(1, 5)]
    want = [dense_greedy(tiny_model, p, 10) for p in prompts]
    cfg = make_config(tiny_model, cache=CacheConfig(block_size=4, num_blocks=128,
                                                    enable_prefix_caching=True))
    engine = InferenceEngine(cfg, model=tiny_model)
    first = engine.generate(prompts[:1], GREEDY)
    rest = engine.generate(prompts[1:], GREEDY)
    assert [o.output_token_ids for o in first + rest] == want
    assert engine.block_manager.stats().prefix_cache_hit_rate > 0.5


def test_abort_mid_generation(tiny_model):
    engine = InferenceEngine(make_config(tiny_model), model=tiny_model)
    rid = engine.add_request([1, 2, 3], SamplingParams(max_tokens=50))
    engine.step()
    engine.step()
    req = engine.scheduler.requests[rid]
    assert req.status == RequestStatus.RUNNING
    assert engine.abort_request(rid)
    assert req.status == RequestStatus.ABORTED
    assert not engine.has_unfinished_requests()
    assert engine.block_manager.num_free_blocks == engine.block_manager.num_blocks


def test_stop_token_and_metrics(tiny_model):
    want = dense_greedy(tiny_model, [7, 7, 7], 10)
    stop = want[3]
    engine = InferenceEngine(make_config(tiny_model), model=tiny_model)
    out = engine.generate([[7, 7, 7]], SamplingParams(max_tokens=10, temperature=0,
                                                       stop_token_ids=[stop]))[0]
    assert out.output_token_ids == want[: want.index(stop) + 1]
    assert out.finish_reason.value == "stop"
    m = engine.stats()["metrics"]
    assert m["requests_finished"] == 1 and m["output_tokens_per_s"] > 0
    assert m["ttft_p50_ms"] > 0


def test_sampling_is_reproducible_with_seed(tiny_model):
    params = SamplingParams(max_tokens=8, temperature=1.0, top_k=20)
    a = InferenceEngine(make_config(tiny_model, seed=3), model=tiny_model).generate([[1, 2]], params)
    b = InferenceEngine(make_config(tiny_model, seed=3), model=tiny_model).generate([[1, 2]], params)
    assert a[0].output_token_ids == b[0].output_token_ids


def test_top_p_top_k_masking():
    from tokenvault.sampling import logits_to_probs

    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    p = logits_to_probs(logits, [SamplingParams(top_k=2)])
    assert (p[0, 2:] == 0).all() and p[0, :2].sum() == pytest.approx(1.0)
    p = logits_to_probs(logits, [SamplingParams(top_p=0.5)])
    assert p[0, 0] == pytest.approx(1.0)
    p = logits_to_probs(logits, [SamplingParams(temperature=0.0)])
    assert p.tolist() == [[1.0, 0.0, 0.0, 0.0, 0.0]]
