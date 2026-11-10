"""Serving metrics: throughput, latency percentiles, KV memory, and $/1M-token cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pulsecache.batching.request import Request
from pulsecache.cache.block_manager import BlockManagerStats


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


@dataclass(frozen=True)
class CostModel:
    """Amortized accelerator cost. Defaults approximate on-demand cloud list prices."""

    hourly_usd: float = 2.0
    name: str = "generic-gpu"

    def cost_per_million_tokens(self, tokens_per_second: float) -> float:
        if tokens_per_second <= 0:
            return float("inf")
        return self.hourly_usd / (tokens_per_second * 3600.0) * 1e6


GPU_PRICES = {
    "a10g": CostModel(1.01, "A10G 24GB"),
    "l4": CostModel(0.80, "L4 24GB"),
    "a100-80g": CostModel(3.67, "A100 80GB"),
    "h100": CostModel(6.98, "H100 80GB"),
    "cpu": CostModel(0.40, "16 vCPU"),
}


@dataclass
class MetricsCollector:
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    num_steps: int = 0
    num_prompt_tokens: int = 0
    num_generated_tokens: int = 0
    num_finished: int = 0
    num_aborted: int = 0
    ttft: list[float] = field(default_factory=list)
    tpot: list[float] = field(default_factory=list)
    e2e: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    batched_tokens: list[int] = field(default_factory=list)
    kv_utilization: list[float] = field(default_factory=list)
    kv_fragmentation: list[float] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def record_step(
        self, batch_size: int, num_tokens: int, step_time: float, kv: BlockManagerStats
    ) -> None:
        self.num_steps += 1
        self.batch_sizes.append(batch_size)
        self.batched_tokens.append(num_tokens)
        self.step_times.append(step_time)
        self.kv_utilization.append(kv.utilization)
        self.kv_fragmentation.append(kv.internal_fragmentation)

    def record_finished(self, req: Request) -> None:
        if req.finish_reason is not None and req.finish_reason.value == "abort":
            self.num_aborted += 1
            return
        self.num_finished += 1
        self.num_prompt_tokens += req.num_prompt_tokens
        self.num_generated_tokens += req.num_output_tokens
        if req.ttft is not None:
            self.ttft.append(req.ttft)
        if req.tpot is not None:
            self.tpot.append(req.tpot)
        if req.e2e_latency is not None:
            self.e2e.append(req.e2e_latency)

    def summary(self, cost: CostModel | None = None) -> dict[str, float]:
        end = self.end_time or time.perf_counter()
        elapsed = max(end - self.start_time, 1e-9)
        gen_tps = self.num_generated_tokens / elapsed
        out = {
            "elapsed_s": elapsed,
            "requests_finished": float(self.num_finished),
            "requests_per_s": self.num_finished / elapsed,
            "generated_tokens": float(self.num_generated_tokens),
            "output_tokens_per_s": gen_tps,
            "total_tokens_per_s": (self.num_generated_tokens + self.num_prompt_tokens) / elapsed,
            "ttft_p50_ms": percentile(self.ttft, 50) * 1e3,
            "ttft_p99_ms": percentile(self.ttft, 99) * 1e3,
            "tpot_p50_ms": percentile(self.tpot, 50) * 1e3,
            "tpot_p99_ms": percentile(self.tpot, 99) * 1e3,
            "e2e_p50_ms": percentile(self.e2e, 50) * 1e3,
            "e2e_p99_ms": percentile(self.e2e, 99) * 1e3,
            "mean_batch_size": float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0,
            "peak_kv_utilization": max(self.kv_utilization, default=0.0),
            "mean_kv_fragmentation": float(np.mean(self.kv_fragmentation))
            if self.kv_fragmentation
            else 0.0,
            "num_steps": float(self.num_steps),
        }
        if cost is not None:
            out["usd_per_1m_output_tokens"] = cost.cost_per_million_tokens(gen_tps)
        return out
