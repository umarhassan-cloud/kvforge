import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from tokenvault import ByteTokenizer, InferenceEngine
from tokenvault.serving import create_app

from .conftest import make_config


@pytest.fixture
def client(tiny_model):
    engine = InferenceEngine(make_config(tiny_model), model=tiny_model, tokenizer=ByteTokenizer())
    with TestClient(create_app(engine, "tiny")) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"


def test_completion(client):
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 5,
                                             "temperature": 0, "ignore_eos": True})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_completion_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True,
        "temperature": 0, "ignore_eos": True,
    }) as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    assert len(lines) >= 2


def test_bad_request_is_400(client):
    r = client.post("/v1/completions", json={"prompt": [99999], "max_tokens": 2})
    assert r.status_code == 400


def test_metrics_endpoint(client):
    client.post("/v1/completions", json={"prompt": "abc", "max_tokens": 3, "ignore_eos": True})
    m = client.get("/metrics").json()
    assert m["metrics"]["requests_finished"] >= 1
    assert "kv_cache" in m
