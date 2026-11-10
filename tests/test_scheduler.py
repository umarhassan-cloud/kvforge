import pytest

from tokenvault.batching.request import FinishReason, Request, RequestStatus
from tokenvault.cache.block_manager import BlockManager
from tokenvault.config import SchedulerConfig, SchedulingPolicy
from tokenvault.sampling import SamplingParams
from tokenvault.scheduler import RequestRejectedError, Scheduler


def make(num_blocks=64, block_size=4, lookahead=0, **cfg) -> Scheduler:
    return Scheduler(SchedulerConfig(**cfg), BlockManager(num_blocks, block_size), lookahead)


def req(rid: str, n: int, arrival: float, priority: int = 0, max_tokens: int = 8) -> Request:
    return Request(rid, list(range(1, n + 1)), SamplingParams(max_tokens=max_tokens),
                   priority=priority, arrival_time=arrival)


def complete(sched_out) -> None:
    """Pretend the model ran: advance computed counts and sample one token if prefill done."""
    for s in sched_out.scheduled:
        r = s.request
        r.num_computed_tokens += s.num_new_tokens
        if r.num_computed_tokens == r.num_tokens:
            r.append_token(0)


def test_fcfs_admission_order_and_max_seqs():
    s = make(max_num_seqs=2)
    for i, t in enumerate([3.0, 1.0, 2.0]):
        s.add_request(req(f"r{i}", 4, arrival=t))
    out = s.schedule()
    assert [x.request.request_id for x in out.scheduled] == ["r1", "r2"]
    assert all(x.is_prefill for x in out.scheduled)
    assert s.num_running == 2 and s.num_waiting == 1


def test_priority_policy():
    s = make(max_num_seqs=1, policy=SchedulingPolicy.PRIORITY)
    s.add_request(req("low", 4, arrival=0.0, priority=5))
    s.add_request(req("high", 4, arrival=1.0, priority=0))
    assert s.schedule().scheduled[0].request.request_id == "high"


def test_prefill_then_decode_phases():
    s = make()
    s.add_request(req("a", 6, 0.0))
    out = s.schedule()
    assert out.scheduled[0].is_prefill and out.scheduled[0].num_new_tokens == 6
    complete(out)
    out = s.schedule()
    assert not out.scheduled[0].is_prefill and out.scheduled[0].num_new_tokens == 1


def test_token_budget_and_chunked_prefill():
    s = make(max_num_batched_tokens=8, enable_chunked_prefill=True)
    s.add_request(req("a", 20, 0.0))
    chunks = []
    while True:
        out = s.schedule()
        x = out.scheduled[0]
        chunks.append(x.num_new_tokens)
        complete(out)
        if not x.is_prefill or x.request.num_output_tokens:
            break
    assert chunks == [8, 8, 4]
    assert out.num_batched_tokens <= 8


def test_decodes_are_scheduled_before_new_prefills():
    s = make(max_num_batched_tokens=10)
    s.add_request(req("a", 4, 0.0))
    complete(s.schedule())
    s.add_request(req("b", 30, 1.0))
    out = s.schedule()
    ids = [(x.request.request_id, x.num_new_tokens) for x in out.scheduled]
    assert ids == [("a", 1), ("b", 9)]  # decode first, prefill chunk fills the rest


def test_no_chunking_rejects_oversized_prompt():
    s = make(max_num_batched_tokens=8, enable_chunked_prefill=False)
    with pytest.raises(RequestRejectedError):
        s.add_request(req("big", 9, 0.0))


def test_rejects_requests_that_can_never_fit():
    s = make(num_blocks=2, block_size=4)
    with pytest.raises(RequestRejectedError):
        s.add_request(req("a", 6, 0.0, max_tokens=8))


def test_preemption_under_memory_pressure_evicts_latest_arrival():
    # 4 blocks x 4 slots. Two requests of 7 tokens use 2 blocks each; decoding to 9 tokens
    # requires a third block for both -> the later arrival is preempted (recompute).
    s = make(num_blocks=4, block_size=4, max_num_seqs=4)
    s.add_request(req("old", 7, 0.0, max_tokens=4))
    s.add_request(req("new", 7, 1.0, max_tokens=4))
    complete(s.schedule())  # both prefilled; each now has 8 tokens, 7 computed
    complete(s.schedule())  # decode: both write slot 7 (still inside block 2)
    out = s.schedule()  # need slot 8 -> new block for each, only 0 free
    assert [r.request_id for r in out.preempted] == ["new"]
    assert [x.request.request_id for x in out.scheduled] == ["old"]
    victim = out.preempted[0]
    assert victim.status == RequestStatus.WAITING and victim.num_computed_tokens == 0
    assert victim.num_preemptions == 1 and s.num_preemptions == 1
    # Once "old" finishes, "new" is re-admitted and recomputes prompt + generated tokens.
    complete(out)
    s.finish_request(s.running[0], FinishReason.LENGTH)
    out = s.schedule()
    assert out.scheduled[0].request is victim
    assert out.scheduled[0].num_new_tokens == victim.num_tokens


def test_lookahead_slots_reserved_for_speculative_decode():
    s = make(num_blocks=64, block_size=4, lookahead=3)
    s.add_request(req("a", 4, 0.0))
    complete(s.schedule())
    out = s.schedule()
    x = out.scheduled[0]
    assert x.num_lookahead_slots == 3 and out.num_batched_tokens == 4
    # 5 tokens + 3 lookahead = 8 slots = 2 blocks
    assert len(s.block_manager.get_block_table("a")) == 2


def test_abort_waiting_and_running():
    s = make(max_num_seqs=1)
    s.add_request(req("a", 4, 0.0))
    s.add_request(req("b", 4, 1.0))
    s.schedule()
    assert s.abort_request("b").status == RequestStatus.ABORTED  # waiting
    assert s.abort_request("a").finish_reason == FinishReason.ABORT  # running
    assert not s.has_unfinished()
    assert s.block_manager.num_free_blocks == s.block_manager.num_blocks
    assert s.abort_request("a") is None
