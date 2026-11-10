import pytest
import torch
from torch import nn

from pulsecache import InferenceEngine, QuantConfig, QuantMode, SamplingParams, build_model
from pulsecache.model_loader import PRESETS
from pulsecache.quant import (
    Int4Linear,
    Int8Linear,
    awq_search_scale,
    dequantize_int4,
    evaluate_perplexity,
    logit_fidelity,
    pack_int4,
    quantize_int4,
    quantize_int8,
    quantize_model,
    unpack_int4,
    weight_memory_bytes,
)
from pulsecache.quant.int4 import resolve_group_size

from .conftest import make_config


def test_int4_pack_unpack_roundtrip():
    q = torch.randint(0, 16, (5, 32), dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.shape == (5, 16) and packed.dtype == torch.uint8
    assert torch.equal(unpack_int4(packed), q)


def test_int8_error_bound():
    w = torch.randn(64, 96)
    q, s = quantize_int8(w)
    assert q.dtype == torch.int8
    err = (q.float() * s[:, None] - w).abs()
    assert (err <= s[:, None] / 2 + 1e-6).all()  # round-to-nearest: |e| <= scale / 2


def test_int4_groupwise_error_bound():
    w = torch.randn(32, 256)
    packed, s, z, g = quantize_int4(w, group_size=64)
    assert g == 64 and s.shape == (32, 4)
    deq = dequantize_int4(packed, s, z, g)
    err = (deq - w).abs().view(32, 4, 64)
    assert (err <= s[..., None] / 2 + 1e-5).all()


def test_group_size_resolution():
    assert resolve_group_size(256, 128) == 128
    assert resolve_group_size(172, 128) == 86
    assert resolve_group_size(64, 128) == 64


@pytest.mark.parametrize("cls,kwargs,tol", [(Int8Linear, {}, 0.02), (Int4Linear, {"group_size": 32}, 0.2)])
def test_quantized_linear_close_to_fp(cls, kwargs, tol):
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=False)
    x = torch.randn(10, 128)
    qlin = cls.from_linear(lin, **kwargs)
    ref = lin(x)
    rel = (qlin(x) - ref).norm() / ref.norm()
    assert rel < tol
    torch.testing.assert_close(qlin(x), x @ qlin.dequantize().T, atol=1e-4, rtol=1e-4)


def test_awq_scale_never_worse_than_rtn_and_helps_with_outlier_channels():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.05
    x = torch.randn(256, 128)
    x[:, :4] *= 30  # salient activation channels, as observed in real LLMs
    fq = lambda t: dequantize_int4(*quantize_int4(t, 32))  # noqa: E731
    scale, rtn, best = awq_search_scale(w, x, fq, grid=20)
    assert best <= rtn
    assert scale is not None and best < 0.9 * rtn
    # Folding the scale into Int4Linear reproduces the searched error.
    lin = nn.Linear(128, 64, bias=False)
    lin.weight.data = w.clone()
    q = Int4Linear.from_linear(lin, 32, input_scale=scale)
    mse = float((q(x) - x @ w.T).pow(2).mean())
    assert mse == pytest.approx(best, rel=1e-3)


@pytest.mark.parametrize("mode", [QuantMode.INT8, QuantMode.INT4])
def test_quantize_model_end_to_end(mode):
    ref = build_model(PRESETS["tiny"], seed=0)
    model = build_model(PRESETS["tiny"], seed=0)
    seqs = torch.randint(0, 512, (4, 24), generator=torch.Generator().manual_seed(0))
    report = quantize_model(model, QuantConfig(mode=mode, group_size=32), list(seqs[:2].split(1)))
    assert report["num_quantized_linears"] == 7 * ref.num_layers
    assert report["weight_bytes_after"] < report["weight_bytes_before"]
    assert report["calib_awq_mse"] <= report["calib_rtn_mse"] + 1e-12
    n_quant = sum(isinstance(m, (Int8Linear, Int4Linear)) for m in model.modules())
    assert n_quant == 7 * ref.num_layers

    fid = logit_fidelity(ref, model, seqs)
    assert fid["top1_agreement"] > (0.95 if mode == QuantMode.INT8 else 0.8)
    ppl_ref, ppl_q = evaluate_perplexity(ref, seqs), evaluate_perplexity(model, seqs)
    assert abs(ppl_q - ppl_ref) / ppl_ref < 0.05


def test_projection_memory_shrinks_by_expected_ratio():
    model = build_model(PRESETS["tiny"], seed=0)
    proj = sum(m.weight.numel() * 4 for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and n != "lm_head")
    before = weight_memory_bytes(model)
    quantize_model(model, QuantConfig(mode=QuantMode.INT8, awq=False))
    saved = before - weight_memory_bytes(model)
    assert saved == pytest.approx(proj * 0.75, rel=0.05)  # fp32 -> int8 (+ per-row scales)


def test_engine_runs_quantized(tiny_model):
    cfg = make_config(tiny_model, quant=QuantConfig(mode=QuantMode.INT4, group_size=32))
    engine = InferenceEngine(cfg, model=tiny_model)
    outs = engine.generate([[1, 2, 3], [4, 5]], SamplingParams(max_tokens=6, temperature=0))
    assert all(len(o.output_token_ids) == 6 for o in outs)
    assert "quant" in engine.stats()
