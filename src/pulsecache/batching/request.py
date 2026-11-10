"""Request state machine.

::

    WAITING ──schedule──▶ RUNNING ──stop/length──▶ FINISHED
       ▲                     │
       └────preempt──────────┘        (any non-final) ──abort──▶ ABORTED

Preemption uses *recompute*: blocks are freed and ``num_computed_tokens`` reset;
on re-admission the prompt **and** already-generated tokens are prefilled again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from pulsecache.sampling import SamplingParams


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"

    @property
    def is_final(self) -> bool:
        return self in (RequestStatus.FINISHED, RequestStatus.ABORTED)


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass(eq=False)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    priority: int = 0  # lower value = more important
    arrival_time: float = field(default_factory=time.perf_counter)
    eos_token_id: int | None = None

    status: RequestStatus = RequestStatus.WAITING
    finish_reason: FinishReason | None = None
    output_token_ids: list[int] = field(default_factory=list)
    # Tokens whose K/V are valid in the target cache. Invariant while RUNNING in decode:
    # num_computed_tokens == num_tokens - 1 (the newest sampled token is not yet fed back).
    num_computed_tokens: int = 0
    # Same, for the draft model's cache during speculative decoding.
    num_draft_computed_tokens: int = 0
    num_preemptions: int = 0

    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    token_times: list[float] = field(default_factory=list)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """True until every token but the newest sampled one is in the cache."""
        return self.num_output_tokens == 0 or self.num_uncomputed_tokens > 1

    @property
    def is_finished(self) -> bool:
        return self.status.is_final

    def append_token(self, token_id: int, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self.output_token_ids.append(token_id)
        self.token_times.append(now)
        if self.first_token_time is None:
            self.first_token_time = now

    def check_stop(self, max_model_len: int) -> FinishReason | None:
        params = self.sampling_params
        last = self.output_token_ids[-1] if self.output_token_ids else None
        if (
            last is not None
            and not params.ignore_eos
            and (last == self.eos_token_id or last in params.stop_token_ids)
        ):
            return FinishReason.STOP
        if self.num_output_tokens >= params.max_tokens or self.num_tokens >= max_model_len:
            return FinishReason.LENGTH
        return None

    # ---- latency metrics
    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def e2e_latency(self) -> float | None:
        return None if self.finish_time is None else self.finish_time - self.arrival_time

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first."""
        if len(self.token_times) < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (len(self.token_times) - 1)


@dataclass
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    new_token_ids: list[int]
    finished: bool
    finish_reason: FinishReason | None = None
    text: str | None = None

    @classmethod
    def from_request(cls, req: Request, new_token_ids: list[int]) -> RequestOutput:
        return cls(
            request_id=req.request_id,
            prompt_token_ids=req.prompt_token_ids,
            output_token_ids=list(req.output_token_ids),
            new_token_ids=new_token_ids,
            finished=req.is_finished,
            finish_reason=req.finish_reason,
        )
