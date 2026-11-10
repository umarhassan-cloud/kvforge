"""Minimal OpenAI-compatible HTTP server.

Endpoints: ``POST /v1/completions``, ``POST /v1/chat/completions`` (both support
``stream=true`` via SSE), ``GET /v1/models``, ``GET /health``, ``GET /metrics``.

Concurrency model: a single background task owns the engine loop and runs each
``engine.step()`` in a worker thread; HTTP handlers only enqueue requests and await
per-request output queues. A lock serializes ``add_request``/``abort`` with ``step``
so the scheduler is never mutated concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pulsecache.batching.request import RequestOutput
from pulsecache.engine import InferenceEngine
from pulsecache.metrics import CostModel
from pulsecache.sampling import SamplingParams
from pulsecache.scheduler import RequestRejectedError


class AsyncEngine:
    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._queues: dict[str, asyncio.Queue[RequestOutput]] = {}
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def stats(self, cost: CostModel | None = None) -> dict[str, Any]:
        with self._lock:
            return self.engine.stats(cost)

    def _step(self) -> list[RequestOutput]:
        with self._lock:
            return self.engine.step()

    async def _loop(self) -> None:
        assert self._wake is not None
        while True:
            if not self.engine.has_unfinished_requests():
                self._wake.clear()
                await self._wake.wait()
                continue
            for out in await asyncio.to_thread(self._step):
                q = self._queues.get(out.request_id)
                if q is not None:
                    q.put_nowait(out)
            await asyncio.sleep(0)

    async def generate(
        self, prompt: str | list[int], params: SamplingParams, priority: int = 0
    ) -> AsyncIterator[RequestOutput]:
        assert self._wake is not None
        request_id = f"cmpl-{uuid.uuid4().hex[:16]}"
        queue: asyncio.Queue[RequestOutput] = asyncio.Queue()
        self._queues[request_id] = queue
        with self._lock:
            self.engine.add_request(prompt, params, request_id=request_id, priority=priority)
        self._wake.set()
        finished = False
        try:
            while not finished:
                out = await queue.get()
                finished = out.finished
                yield out
        finally:
            self._queues.pop(request_id, None)
            if not finished:  # client disconnected
                with self._lock:
                    self.engine.abort_request(request_id)


class _GenParams(BaseModel):
    model: str = "pulsecache"
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    priority: int = 0
    ignore_eos: bool = False

    def sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p,
            top_k=self.top_k, ignore_eos=self.ignore_eos,
        )


class CompletionRequest(_GenParams):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_GenParams):
    messages: list[ChatMessage]


def render_chat(messages: list[ChatMessage]) -> str:
    return "".join(f"<|{m.role}|>\n{m.content}\n" for m in messages) + "<|assistant|>\n"


def _usage(out: RequestOutput) -> dict[str, int]:
    p, c = len(out.prompt_token_ids), len(out.output_token_ids)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def create_app(
    engine: InferenceEngine, model_name: str = "pulsecache", cost: CostModel | None = None
) -> FastAPI:
    async_engine = AsyncEngine(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await async_engine.start()
        yield
        await async_engine.stop()

    app = FastAPI(title="pulsecache", version="0.1.0", lifespan=lifespan)
    app.state.engine = async_engine

    def decode(ids: list[int]) -> str:
        return engine.tokenizer.decode(ids) if engine.tokenizer else " ".join(map(str, ids))

    async def run(
        prompt: str | list[int], req: _GenParams, chat: bool
    ) -> dict[str, Any] | StreamingResponse:
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        try:
            stream = async_engine.generate(prompt, req.sampling_params(), req.priority)
            first = await stream.__anext__()
        except (RequestRejectedError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                out: RequestOutput | None = first
                sent = 0
                while out is not None:
                    text = decode(out.output_token_ids)
                    delta, sent = text[sent:], len(text)
                    reason = out.finish_reason.value if out.finish_reason else None
                    choice: dict[str, Any] = (
                        {"index": 0, "delta": {"content": delta}, "finish_reason": reason}
                        if chat else {"index": 0, "text": delta, "finish_reason": reason}
                    )
                    chunk = {"id": out.request_id, "object": f"{obj}.chunk", "created": created,
                             "model": model_name, "choices": [choice]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    out = None if out.finished else await stream.__anext__()
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        out = first
        async for out in stream:  # noqa: B007 - drain to the final output
            pass
        text = decode(out.output_token_ids)
        reason = out.finish_reason.value if out.finish_reason else None
        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
            if chat else {"index": 0, "text": text, "logprobs": None, "finish_reason": reason}
        )
        return {"id": out.request_id, "object": obj, "created": created, "model": model_name,
                "choices": [choice], "usage": _usage(out)}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "pulsecache"}]}

    @app.post("/v1/completions", response_model=None)
    async def completions(req: CompletionRequest) -> dict[str, Any] | StreamingResponse:
        return await run(req.prompt, req, chat=False)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
        if engine.tokenizer is None:
            raise HTTPException(status_code=400, detail="chat requires a tokenizer")
        return await run(render_chat(req.messages), req, chat=True)

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return async_engine.stats(cost)

    return app
