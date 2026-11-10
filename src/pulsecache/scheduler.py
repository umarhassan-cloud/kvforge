"""Iteration-level scheduler (Orca-style continuous batching over a paged KV cache).

Every engine step the scheduler re-decides the batch under two budgets:

* ``max_num_batched_tokens`` — compute budget per forward (prefill chunks + decodes),
* ``max_num_seqs``           — concurrency cap,

plus the memory constraint that every scheduled token has a KV slot.

Order of work in :meth:`Scheduler.schedule`:

1. **Running requests first** (in policy order). Decodes cost ``1 + lookahead`` tokens;
   in-flight chunked prefills take what is left of the budget. If the block pool cannot
   supply slots, the *lowest-priority* running request is preempted (recompute) until
   it can — so the most important requests always make progress.
2. **Admit waiting requests** only if nothing was preempted this step (otherwise we
   would thrash). Admission is head-of-line ordered: if the head does not fit, stop,
   which keeps FCFS/priority fairness instead of letting short prompts jump the queue.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

from pulsecache.batching.request import FinishReason, Request, RequestStatus
from pulsecache.cache.block_manager import BlockManager
from pulsecache.config import SchedulerConfig, SchedulingPolicy


@dataclass
class ScheduledRequest:
    request: Request
    num_new_tokens: int
    is_prefill: bool
    num_lookahead_slots: int = 0


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)
    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled

    @property
    def num_batched_tokens(self) -> int:
        return sum(s.num_new_tokens + s.num_lookahead_slots for s in self.scheduled)

    @property
    def prefills(self) -> list[ScheduledRequest]:
        return [s for s in self.scheduled if s.is_prefill]

    @property
    def decodes(self) -> list[ScheduledRequest]:
        return [s for s in self.scheduled if not s.is_prefill]


class RequestRejectedError(ValueError):
    pass


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: BlockManager,
        num_lookahead_slots: int = 0,
    ) -> None:
        self.config = config
        self.block_manager = block_manager
        self.num_lookahead_slots = num_lookahead_slots
        self._waiting: list[tuple[tuple, Request]] = []
        self._counter = itertools.count()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}
        self.num_preemptions = 0

    # ------------------------------------------------------------- bookkeeping
    def _key(self, req: Request) -> tuple:
        if self.config.policy == SchedulingPolicy.PRIORITY:
            return (req.priority, req.arrival_time)
        return (req.arrival_time,)

    def _push_waiting(self, req: Request) -> None:
        heapq.heappush(self._waiting, (self._key(req) + (next(self._counter),), req))

    def _peek_waiting(self) -> Request | None:
        while self._waiting and self._waiting[0][1].status != RequestStatus.WAITING:
            heapq.heappop(self._waiting)  # lazily drop aborted entries
        return self._waiting[0][1] if self._waiting else None

    @property
    def num_waiting(self) -> int:
        return sum(1 for _, r in self._waiting if r.status == RequestStatus.WAITING)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_unfinished(self) -> bool:
        return bool(self.running) or self._peek_waiting() is not None

    def add_request(self, req: Request) -> None:
        if req.request_id in self.requests:
            raise RequestRejectedError(f"duplicate request id {req.request_id}")
        if req.num_prompt_tokens == 0:
            raise RequestRejectedError("empty prompt")
        if req.num_prompt_tokens >= self.config.max_model_len:
            raise RequestRejectedError(
                f"prompt of {req.num_prompt_tokens} tokens exceeds max_model_len "
                f"{self.config.max_model_len}"
            )
        if (
            not self.config.enable_chunked_prefill
            and req.num_prompt_tokens > self.config.max_num_batched_tokens
        ):
            raise RequestRejectedError("prompt exceeds max_num_batched_tokens without chunked prefill")
        worst_case = min(
            req.num_prompt_tokens + req.sampling_params.max_tokens, self.config.max_model_len
        ) + self.num_lookahead_slots
        bm = self.block_manager
        if bm.blocks_needed(worst_case) > bm.num_blocks:
            raise RequestRejectedError("request can never fit in the KV cache")
        req.status = RequestStatus.WAITING
        self.requests[req.request_id] = req
        self._push_waiting(req)

    def _preempt(self, req: Request) -> None:
        self.block_manager.free(req.request_id)
        req.status = RequestStatus.WAITING
        req.num_computed_tokens = 0
        req.num_draft_computed_tokens = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        self._push_waiting(req)

    def finish_request(self, req: Request, reason: FinishReason) -> None:
        req.status = RequestStatus.ABORTED if reason == FinishReason.ABORT else RequestStatus.FINISHED
        req.finish_reason = reason
        req.finish_time = time.perf_counter()
        if req in self.running:
            self.running.remove(req)
        self.block_manager.free(req.request_id)
        self.requests.pop(req.request_id, None)

    def abort_request(self, request_id: str) -> Request | None:
        req = self.requests.get(request_id)
        if req is None or req.is_finished:
            return None
        self.finish_request(req, FinishReason.ABORT)
        return req

    # ---------------------------------------------------------------- schedule
    def schedule(self) -> SchedulerOutput:
        cfg, bm = self.config, self.block_manager
        out = SchedulerOutput()
        budget = cfg.max_num_batched_tokens

        queue = sorted(self.running, key=self._key)
        still_running: list[Request] = []
        while queue:
            req = queue.pop(0)
            prefill = req.is_prefill
            n_new = req.num_uncomputed_tokens
            lookahead = 0 if prefill else self.num_lookahead_slots
            if prefill and cfg.enable_chunked_prefill:
                n_new = min(n_new, budget)
            if n_new <= 0 or n_new + lookahead > budget:
                still_running.append(req)
                continue

            total = req.num_computed_tokens + n_new + lookahead
            fits = True
            while not bm.can_append_slots(req.request_id, total, req.num_computed_tokens):
                if not queue:
                    fits = False
                    break
                victim = queue.pop()  # lowest priority / latest arrival
                self._preempt(victim)
                out.preempted.append(victim)
            if not fits:
                self._preempt(req)
                out.preempted.append(req)
                continue

            out.blocks_to_copy += bm.append_slots(req.request_id, total, req.num_computed_tokens)
            out.scheduled.append(ScheduledRequest(req, n_new, prefill, lookahead))
            still_running.append(req)
            budget -= n_new + lookahead
        self.running = still_running

        if out.preempted:
            return out

        now = time.perf_counter()
        while budget > 0 and len(self.running) < cfg.max_num_seqs:
            req = self._peek_waiting()
            if req is None:
                break
            tokens = req.all_token_ids
            if not cfg.enable_chunked_prefill and len(tokens) > budget:
                break
            if not bm.can_allocate(tokens):
                break
            heapq.heappop(self._waiting)
            num_cached = bm.allocate(req.request_id, tokens)
            req.num_computed_tokens = num_cached
            req.num_draft_computed_tokens = num_cached
            n_new = len(tokens) - num_cached
            if cfg.enable_chunked_prefill:
                n_new = min(n_new, budget)
            req.status = RequestStatus.RUNNING
            if req.first_scheduled_time is None:
                req.first_scheduled_time = now
            out.scheduled.append(ScheduledRequest(req, n_new, True, 0))
            self.running.append(req)
            budget -= n_new
        return out
