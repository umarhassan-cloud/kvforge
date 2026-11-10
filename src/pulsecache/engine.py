"""InferenceEngine: wires model, paged KV cache, scheduler, speculative decoding and
quantization into a single ``step()`` loop.

One ``step()`` = one scheduler decision + at most two forward groups:

* **normal group** — prefill chunks (and plain decodes when speculation is off): one
  varlen target forward; the draft (if any) consumes the same prefill tokens so its
  cache stays in sync;
* **speculative group** — decoding sequences: k draft forwards + one target verify.

The engine is synchronous and single-threaded by design; :mod:`pulsecache.serving.api`
drives it from a background task.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import torch

from pulsecache.batching.continuous import BatchItem, ModelRunner
from pulsecache.batching.request import FinishReason, Request, RequestOutput
from pulsecache.cache.block_manager import BlockManager
from pulsecache.cache.kv_cache import PagedKVCache
from pulsecache.config import EngineConfig, QuantMode
from pulsecache.metrics import CostModel, MetricsCollector
from pulsecache.model import Transformer
from pulsecache.model_loader import Tokenizer, build_model
from pulsecache.quant.calibrate import quantize_model, weight_memory_bytes
from pulsecache.sampling import Sampler, SamplingParams
from pulsecache.scheduler import ScheduledRequest, Scheduler
from pulsecache.speculative.draft_verify import SpeculativeDecoder, make_layer_skip_draft


class InferenceEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        model: Transformer | None = None,
        draft_model: Transformer | None = None,
        tokenizer: Tokenizer | None = None,
        calib_batches: list[torch.Tensor] | None = None,
    ) -> None:
        self.config = config = config or EngineConfig()
        self.device = torch.device(config.device)
        self.tokenizer = tokenizer

        self.model = model if model is not None else build_model(
            config.model, self.device, config.dtype, config.seed
        )
        config.model = self.model.config
        self.model.set_attention_backend(config.attention_backend)

        self.quant_report: dict[str, float] = {}
        if config.quant.mode != QuantMode.NONE:
            self.quant_report = quantize_model(self.model, config.quant, calib_batches)

        mcfg, ccfg = self.model.config, config.cache
        assert mcfg.head_dim is not None
        num_blocks = ccfg.resolve_num_blocks(mcfg)
        self.block_manager = BlockManager(
            num_blocks, ccfg.block_size, ccfg.enable_prefix_caching, ccfg.watermark
        )
        self.kv_cache = PagedKVCache(
            mcfg.num_layers, num_blocks, ccfg.block_size, mcfg.num_kv_heads, mcfg.head_dim,
            config.dtype, self.device,
        )
        self.runner = ModelRunner(self.model, self.kv_cache, self.block_manager, self.device)
        self.sampler = Sampler(config.seed, self.device)

        self.spec: SpeculativeDecoder | None = None
        self.draft_kv_cache: PagedKVCache | None = None
        self.draft_runner: ModelRunner | None = None
        lookahead = 0
        if config.speculative is not None:
            scfg = config.speculative
            if draft_model is None:
                draft_model = make_layer_skip_draft(self.model, scfg.draft_num_layers or 1)
            dcfg = draft_model.config
            if dcfg.vocab_size != mcfg.vocab_size:
                raise ValueError("draft and target must share a vocabulary")
            assert dcfg.head_dim is not None
            draft_model.set_attention_backend(config.attention_backend)
            # Same block tables as the target (shared BlockManager), separate KV tensors.
            self.draft_kv_cache = PagedKVCache(
                dcfg.num_layers, num_blocks, ccfg.block_size, dcfg.num_kv_heads, dcfg.head_dim,
                config.dtype, self.device,
            )
            self.draft_runner = ModelRunner(
                draft_model, self.draft_kv_cache, self.block_manager, self.device
            )
            self.spec = SpeculativeDecoder(
                self.draft_runner, self.runner, self.sampler, scfg.num_speculative_tokens
            )
            lookahead = scfg.num_speculative_tokens

        self.scheduler = Scheduler(config.scheduler, self.block_manager, lookahead)
        self.metrics = MetricsCollector()
        self._ids = itertools.count()

    # ------------------------------------------------------------------ API
    def add_request(
        self,
        prompt: str | Sequence[int],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        priority: int = 0,
        arrival_time: float | None = None,
    ) -> str:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("string prompts need a tokenizer")
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)
        if any(t < 0 or t >= self.model.config.vocab_size for t in token_ids):
            raise ValueError("prompt contains out-of-vocabulary token ids")
        request_id = request_id or f"req-{next(self._ids)}"
        eos = self.tokenizer.eos_token_id if self.tokenizer else self.model.config.eos_token_id
        req = Request(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params or SamplingParams(),
            priority=priority,
            eos_token_id=eos,
        )
        if arrival_time is not None:
            req.arrival_time = arrival_time
        self.scheduler.add_request(req)
        return request_id

    def abort_request(self, request_id: str) -> bool:
        req = self.scheduler.abort_request(request_id)
        if req is not None:
            self.metrics.record_finished(req)
        return req is not None

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def generate(
        self,
        prompts: Sequence[str | Sequence[int]],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
    ) -> list[RequestOutput]:
        """Offline batch generation; returns outputs in prompt order."""
        if sampling_params is None or isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params)]
        done: dict[str, RequestOutput] = {}
        while self.has_unfinished_requests():
            for out in self.step():
                if out.finished:
                    done[out.request_id] = out
        return [done[i] for i in ids if i in done]

    # ----------------------------------------------------------------- step
    def step(self) -> list[RequestOutput]:
        t0 = time.perf_counter()
        sched = self.scheduler.schedule()
        if sched.is_empty:
            return []
        self.kv_cache.copy_blocks(sched.blocks_to_copy)
        if self.draft_kv_cache is not None:
            self.draft_kv_cache.copy_blocks(sched.blocks_to_copy)

        normal: list[ScheduledRequest] = []
        spec: list[ScheduledRequest] = []
        for s in sched.scheduled:
            (spec if (self.spec is not None and not s.is_prefill) else normal).append(s)
        k = 0
        if spec:
            assert self.spec is not None
            max_len = self.config.scheduler.max_model_len
            k = min(self.spec.k, min(max_len - s.request.num_tokens for s in spec))
            if k < 1:
                normal, spec = normal + spec, []

        new_tokens: dict[str, list[int]] = {}
        if normal:
            self._run_normal(normal, new_tokens)
        if spec:
            self._run_speculative(spec, k, new_tokens)

        outputs = self._postprocess(sched.scheduled, new_tokens)
        self.metrics.record_step(
            len(sched.scheduled), sched.num_batched_tokens, time.perf_counter() - t0,
            self.block_manager.stats(),
        )
        return outputs

    def _run_normal(self, group: list[ScheduledRequest], new_tokens: dict[str, list[int]]) -> None:
        items: list[BatchItem] = []
        sampled: list[Request] = []
        for s in group:
            r = s.request
            start = r.num_computed_tokens
            completes = start + s.num_new_tokens == r.num_tokens
            items.append(BatchItem(
                r.request_id, r.all_token_ids[start : start + s.num_new_tokens], start,
                "last" if completes else "none",
            ))
            if completes:
                sampled.append(r)
        logits = self.runner.execute(items)

        if self.draft_runner is not None:
            in_sync = [
                (it, s.request) for it, s in zip(items, group)
                if s.request.num_draft_computed_tokens == it.start_pos
            ]
            if in_sync:
                self.draft_runner.execute(
                    [BatchItem(it.seq_id, it.token_ids, it.start_pos, "none") for it, _ in in_sync]
                )
                for it, r in in_sync:
                    r.num_draft_computed_tokens += len(it.token_ids)

        for s in group:
            s.request.num_computed_tokens += s.num_new_tokens
        if sampled:
            tokens, _ = self.sampler(logits, [r.sampling_params for r in sampled])
            for r, t in zip(sampled, tokens.tolist()):
                new_tokens[r.request_id] = [t]

    def _run_speculative(
        self, group: list[ScheduledRequest], k: int, new_tokens: dict[str, list[int]]
    ) -> None:
        assert self.spec is not None
        reqs = [s.request for s in group]
        n_before = [r.num_tokens for r in reqs]
        result = self.spec.step(reqs, k)
        for i, r in enumerate(reqs):
            # x_last plus the accepted drafts now have valid target K/V.
            r.num_computed_tokens = n_before[i] + result.num_accepted[i]
            r.num_draft_computed_tokens = result.draft_computed[i]
            new_tokens[r.request_id] = result.new_tokens[i]

    def _postprocess(
        self, scheduled: list[ScheduledRequest], new_tokens: dict[str, list[int]]
    ) -> list[RequestOutput]:
        now = time.perf_counter()
        max_len = self.config.scheduler.max_model_len
        outputs: list[RequestOutput] = []
        for s in scheduled:
            r = s.request
            emitted: list[int] = []
            reason: FinishReason | None = None
            for t in new_tokens.get(r.request_id, []):
                r.append_token(t, now)
                emitted.append(t)
                reason = r.check_stop(max_len)
                if reason is not None:
                    break
            if reason is not None:
                self.scheduler.finish_request(r, reason)
                self.metrics.record_finished(r)
            else:
                computed = r.num_computed_tokens
                if self.spec is not None:
                    computed = min(computed, r.num_draft_computed_tokens)
                self.block_manager.mark_computed(r.request_id, r.all_token_ids, computed)
            if emitted or r.is_finished:
                out = RequestOutput.from_request(r, emitted)
                if self.tokenizer is not None:
                    out.text = self.tokenizer.decode(out.output_token_ids)
                outputs.append(out)
        return outputs

    # ---------------------------------------------------------------- stats
    def stats(self, cost: CostModel | None = None) -> dict[str, Any]:
        bm = self.block_manager.stats()
        out: dict[str, Any] = {
            "metrics": self.metrics.summary(cost),
            "kv_cache": {
                **asdict(bm),
                "bytes_total": self.kv_cache.memory_bytes,
                "bytes_per_block": self.kv_cache.bytes_per_block,
                "bytes_in_use": bm.num_used_blocks * self.kv_cache.bytes_per_block,
            },
            "scheduler": {
                "num_running": self.scheduler.num_running,
                "num_waiting": self.scheduler.num_waiting,
                "num_preemptions": self.scheduler.num_preemptions,
            },
            "weights_bytes": weight_memory_bytes(self.model),
            "forward_passes": self.runner.num_forward_passes,
        }
        if self.quant_report:
            out["quant"] = self.quant_report
        if self.spec is not None:
            out["speculative"] = self.spec.stats.as_dict()
        return out
