"""Reproducible kvforge benchmark suite.

Modes (same weights, same workload, same greedy decoding):

    naive        HF-style static batching + padded contiguous KV (benchmarks/baselines.py)
    continuous   kvforge paged KV + iteration-level continuous batching
    speculative  continuous + draft/verify speculative decoding (layer-skip self-draft)
    int8 / int4  continuous + weight-only quantization (AWQ-calibrated)

Extra experiments:

    --spec-sweep       acceptance rate vs speedup at batch size 1 across draft/target alignment
    --quant-accuracy   perplexity / top-1 agreement / KL for fp32 vs INT8 vs INT4
    --roofline         analytic GPU projection for Llama-3.1-8B (clearly labelled, NOT measured)

Usage::

    python benchmarks/run_bench.py                     # everything, ~2-4 min on a laptop CPU
    python benchmarks/run_bench.py --quick             # CI-sized smoke run
    python benchmarks/run_bench.py --device cuda --preset llama-1b --num-requests 256

Results are printed as Markdown and written to ``benchmarks/results/<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "src"))

from baselines import BaselineRequest, naive_static_generate  # noqa: E402
from kvforge import (  # noqa: E402
    PRESETS,
    CacheConfig,
    EngineConfig,
    InferenceEngine,
    QuantConfig,
    QuantMode,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
    Transformer,
    build_model,
)
from kvforge.metrics import GPU_PRICES, CostModel, percentile  # noqa: E402
from kvforge.quant import evaluate_perplexity, logit_fidelity, quantize_model, weight_memory_bytes  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class Workload:
    prompts: list[list[int]]
    output_lens: list[int]

    @property
    def total_output_tokens(self) -> int:
        return sum(self.output_lens)


def make_workload(n: int, vocab: int, prompt: tuple[int, int], output: tuple[int, int], seed: int) -> Workload:
    rng = random.Random(seed)
    prompts = [[rng.randrange(1, vocab) for _ in range(rng.randint(*prompt))] for _ in range(n)]
    return Workload(prompts, [rng.randint(*output) for _ in range(n)])


LM_HEAD_SHARPEN = 8.0


def build_target(preset: str, device: str, dtype: torch.dtype, residual_scale: float,
                 draft_layers: int, seed: int) -> Transformer:
    """Random-weight stand-in for a trained LM, with two documented adjustments:

    * the LM head is scaled by ``LM_HEAD_SHARPEN`` so next-token entropy is a few nats
      (like a trained model) rather than ~log(vocab) (uniform), which would make
      perplexity and acceptance-rate measurements meaningless;
    * layers above ``draft_layers`` have their residual writes damped by
      ``residual_scale``. Trained LLMs have strongly correlated early-exit and final
      predictions; random weights do not. Damping emulates that correlation in a
      controlled, tunable way (scale=1.0 -> plain random model).
    """
    model = build_model(preset, device, dtype, seed)
    with torch.no_grad():
        model.lm_head.weight.mul_(LM_HEAD_SHARPEN)
        for layer in model.layers[draft_layers:]:
            layer.self_attn.o_proj.weight.mul_(residual_scale)
            layer.mlp.down_proj.weight.mul_(residual_scale)
    return model


def engine_config(args: argparse.Namespace, model: Transformer, *, max_num_seqs: int | None = None,
                  spec_k: int = 0, quant: QuantMode = QuantMode.NONE) -> EngineConfig:
    return EngineConfig(
        model=model.config,
        cache=CacheConfig(block_size=args.block_size, num_blocks=args.num_blocks),
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs or args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=model.config.max_position_embeddings,
        ),
        speculative=SpeculativeConfig(spec_k, args.draft_layers) if spec_k else None,
        quant=QuantConfig(mode=quant, group_size=args.group_size),
        device=args.device,
        dtype=DTYPES[args.dtype],
        seed=args.seed,
    )


def params_for(n: int) -> SamplingParams:
    return SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)


def run_engine(engine: InferenceEngine, wl: Workload, cost: CostModel) -> dict:
    # Warm-up (allocator, first-call kernel selection) then reset counters.
    engine.generate([wl.prompts[0][:8]], params_for(4))
    engine.metrics.reset()
    engine.runner.num_forward_passes = 0
    if engine.spec is not None:
        from kvforge.speculative import SpecDecodeStats

        engine.spec.stats = SpecDecodeStats()
    peak_blocks = 0
    t0 = time.perf_counter()
    for p, n in zip(wl.prompts, wl.output_lens):
        engine.add_request(p, params_for(n), arrival_time=t0)
    engine.metrics.start_time = t0
    while engine.has_unfinished_requests():
        engine.step()
        peak_blocks = max(peak_blocks, engine.block_manager.stats().num_used_blocks)
    engine.metrics.end_time = time.perf_counter()
    stats = engine.stats(cost)
    m = stats["metrics"]
    row = {
        "output_tok_s": m["output_tokens_per_s"],
        "ttft_p50_ms": m["ttft_p50_ms"],
        "ttft_p99_ms": m["ttft_p99_ms"],
        "tpot_p50_ms": m["tpot_p50_ms"],
        "tpot_p99_ms": m["tpot_p99_ms"],
        "e2e_p50_ms": m["e2e_p50_ms"],
        "e2e_p99_ms": m["e2e_p99_ms"],
        "mean_batch": m["mean_batch_size"],
        "peak_kv_mib": peak_blocks * engine.kv_cache.bytes_per_block / 2**20,
        "mean_kv_frag": m["mean_kv_fragmentation"],
        "usd_per_1m": m["usd_per_1m_output_tokens"],
        "weights_mib": weight_memory_bytes(engine.model) / 2**20,
        "forward_passes": stats["forward_passes"],
        "preemptions": stats["scheduler"]["num_preemptions"],
        "generated": m["generated_tokens"],
    }
    if "speculative" in stats:
        row["acceptance_rate"] = stats["speculative"]["acceptance_rate"]
        row["per_token_acceptance"] = stats["speculative"]["per_token_acceptance"]
        row["tokens_per_step"] = stats["speculative"]["mean_tokens_per_step"]
    return row


def run_naive(model: Transformer, wl: Workload, batch_size: int, cost: CostModel) -> dict:
    naive_static_generate(model, [BaselineRequest(wl.prompts[0][:8], params_for(4))], 1)  # warm-up
    reqs = [BaselineRequest(p, params_for(n)) for p, n in zip(wl.prompts, wl.output_lens)]
    res = naive_static_generate(model, reqs, batch_size)
    ttft = [r.first_token_time - r.arrival_time for r in reqs if r.first_token_time]
    e2e = [r.finish_time - r.arrival_time for r in reqs if r.finish_time]
    tpot = [(r.finish_time - r.first_token_time) / max(1, len(r.output_token_ids) - 1)
            for r in reqs if r.finish_time and r.first_token_time]
    gen = sum(len(r.output_token_ids) for r in reqs)
    tps = gen / res.elapsed_s
    return {
        "output_tok_s": tps,
        "ttft_p50_ms": percentile(ttft, 50) * 1e3,
        "ttft_p99_ms": percentile(ttft, 99) * 1e3,
        "tpot_p50_ms": percentile(tpot, 50) * 1e3,
        "tpot_p99_ms": percentile(tpot, 99) * 1e3,
        "e2e_p50_ms": percentile(e2e, 50) * 1e3,
        "e2e_p99_ms": percentile(e2e, 99) * 1e3,
        "mean_batch": float(batch_size),
        "peak_kv_mib": res.peak_kv_bytes / 2**20,
        "mean_kv_frag": res.wasted_token_steps / max(1, res.total_token_steps),
        "usd_per_1m": cost.cost_per_million_tokens(tps),
        "weights_mib": weight_memory_bytes(model) / 2**20,
        "generated": float(gen),
    }


def bench_modes(args: argparse.Namespace, cost: CostModel) -> dict[str, dict]:
    dtype = DTYPES[args.dtype]
    factory: Callable[[], Transformer] = lambda: build_target(  # noqa: E731
        args.preset, args.device, dtype, args.residual_scale, args.draft_layers, args.seed)
    vocab = PRESETS[args.preset].vocab_size
    wl = make_workload(args.num_requests, vocab, tuple(args.prompt_len), tuple(args.output_len), args.seed)
    calib = [torch.tensor(wl.prompts[i][:32])[None] for i in range(min(8, len(wl.prompts)))]

    results: dict[str, dict] = {}
    for mode in args.modes:
        model = factory()
        print(f"  running {mode} ...", flush=True)
        if mode == "naive":
            results[mode] = run_naive(model, wl, args.max_num_seqs, cost)
            continue
        if mode == "continuous":
            cfg = engine_config(args, model)
        elif mode == "speculative":
            cfg = engine_config(args, model, spec_k=args.spec_k)
        elif mode in ("int8", "int4"):
            cfg = engine_config(args, model, quant=QuantMode(mode))
        else:
            raise ValueError(mode)
        engine = InferenceEngine(cfg, model=model, calib_batches=calib)
        results[mode] = run_engine(engine, wl, cost)
    return results


def spec_sweep(args: argparse.Namespace, cost: CostModel) -> list[dict]:
    dtype = DTYPES[args.dtype]
    vocab = PRESETS[args.preset].vocab_size
    wl = make_workload(args.sweep_requests, vocab, (32, 64), (64, 64), args.seed + 1)
    rows = []
    for scale in args.sweep_scales:
        model = build_target(args.preset, args.device, dtype, scale, args.draft_layers, args.seed)
        ref = run_engine(InferenceEngine(engine_config(args, model, max_num_seqs=1), model=model), wl, cost)
        for k in args.sweep_k:
            model = build_target(args.preset, args.device, dtype, scale, args.draft_layers, args.seed)
            cfg = engine_config(args, model, max_num_seqs=1, spec_k=k)
            spec = run_engine(InferenceEngine(cfg, model=model), wl, cost)
            a = spec["per_token_acceptance"]
            rows.append({
                "residual_scale": scale,
                "k": k,
                "acceptance_rate": spec["acceptance_rate"],
                "alpha": a,
                "tokens_per_step": spec["tokens_per_step"],
                "theory_tokens_per_step": (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1,
                "baseline_tpot_ms": ref["tpot_p50_ms"],
                "spec_tpot_ms": spec["tpot_p50_ms"],
                "speedup": ref["e2e_p50_ms"] / spec["e2e_p50_ms"],
            })
            print(f"  sweep scale={scale} k={k}: alpha={a:.2f} speedup={rows[-1]['speedup']:.2f}x", flush=True)
    return rows


@torch.inference_mode()
def quant_accuracy(args: argparse.Namespace) -> list[dict]:
    """Perplexity on sequences *sampled from the fp32 model itself* (the only honest
    'held-out text' a random-weight model has), plus top-1 agreement and KL."""
    dtype = DTYPES[args.dtype]
    ref = build_target(args.preset, args.device, dtype, args.residual_scale, args.draft_layers, args.seed)
    eng = InferenceEngine(engine_config(args, ref), model=ref)
    vocab = ref.config.vocab_size
    rng = random.Random(args.seed + 7)
    prompts = [[rng.randrange(1, vocab) for _ in range(8)] for _ in range(24)]
    outs = eng.generate(prompts, SamplingParams(max_tokens=120, temperature=0.8, ignore_eos=True))
    seqs = torch.tensor([o.prompt_token_ids + o.output_token_ids for o in outs])
    calib, held_out = list(seqs[:8].split(1)), seqs[8:]

    rows = [{"mode": "fp32", "ppl": evaluate_perplexity(ref, held_out), "top1_agreement": 1.0,
             "kl_divergence": 0.0, "weights_mib": weight_memory_bytes(ref) / 2**20}]
    variants = [(QuantMode.INT8, False, 0), (QuantMode.INT8, True, 0)] + [
        (QuantMode.INT4, awq, g) for g in sorted({args.group_size, 32}, reverse=True) for awq in (False, True)
    ]
    for mode, awq, g in variants:
        m = build_target(args.preset, args.device, dtype, args.residual_scale, args.draft_layers, args.seed)
        quantize_model(m, QuantConfig(mode=mode, group_size=g or args.group_size, awq=awq), calib)
        fid = logit_fidelity(ref, m, held_out)
        name = f"{mode.value}{f'-g{g}' if mode == QuantMode.INT4 else ''}-{'awq' if awq else 'rtn'}"
        rows.append({"mode": name, "ppl": evaluate_perplexity(m, held_out),
                     **fid, "weights_mib": weight_memory_bytes(m) / 2**20})
    return rows


def roofline(batch: int = 32, context: int = 1024) -> list[dict]:
    """Analytic decode-throughput projection for a Llama-3.1-8B-shaped model in fp16.

    Decode is HBM-bandwidth bound: every step streams all weights once plus each
    sequence's KV. ``tokens/s = batch * efficiency * BW / (W + batch * ctx * kv_per_token)``.
    The efficiency factors are *assumptions* (see BENCHMARK_REPORT.md), not measurements.
    """
    cfg = PRESETS["llama-8b"]
    assert cfg.head_dim is not None
    kv_tok = 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * 2
    n = cfg.num_params
    # HBM bytes moved per decode step for the weights. Fused kernels (vLLM: Marlin / CUTLASS)
    # read the packed weights once. kvforge's reference path reads the packed weights, writes
    # an fp16 temporary, then the GEMM reads it back: packed + 2 + 2 bytes per parameter.
    weight_bytes = {
        "vllm (fused CUDA kernels)": {"fp16": 2.0 * n, "int8": 1.0 * n, "int4": 0.56 * n},
        "kvforge (torch reference path)": {"fp16": 2.0 * n, "int8": 5.0 * n, "int4": 4.56 * n},
    }
    gpus = {"L4": (300e9, GPU_PRICES["l4"]), "A100-80G": (2.0e12, GPU_PRICES["a100-80g"]),
            "H100": (3.35e12, GPU_PRICES["h100"])}
    eff = {"vllm (fused CUDA kernels)": 0.80, "kvforge (torch reference path)": 0.52}
    rows = []
    for gpu, (bw, price) in gpus.items():
        for wname in ("fp16", "int8", "int4"):
            for impl, e in eff.items():
                tps = batch * e * bw / (weight_bytes[impl][wname] + batch * context * kv_tok)
                rows.append({"gpu": gpu, "weights": wname, "impl": impl, "tok_s": tps,
                             "usd_per_1m": price.cost_per_million_tokens(tps)})
    return rows


def md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---:" if c[2] else "---" for c in cols) + "|"
    body = []
    for r in rows:
        cells = []
        for key, _, fmt in cols:
            v = r.get(key, "")
            cells.append(format(v, fmt) if fmt and isinstance(v, (int, float)) else str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--modes", nargs="+", default=["naive", "continuous", "speculative", "int8", "int4"])
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--prompt-len", type=int, nargs=2, default=[16, 192])
    ap.add_argument("--output-len", type=int, nargs=2, default=[8, 160])
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=1024)
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--draft-layers", type=int, default=1)
    ap.add_argument("--residual-scale", type=float, default=0.05,
                    help="damping of layers above the draft; 0.05 -> ~64%% greedy draft agreement on 'small'")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--price", default="cpu", choices=sorted(GPU_PRICES))
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-quant-accuracy", action="store_true")
    ap.add_argument("--sweep-scales", type=float, nargs="+", default=[0.15, 0.07, 0.05, 0.03])
    ap.add_argument("--sweep-k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--sweep-requests", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None,
                    help="torch intra-op threads (small CPU ops are often faster with fewer)")
    ap.add_argument("--quick", action="store_true", help="tiny model, few requests (CI smoke)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.quick:
        args.preset, args.num_requests, args.output_len = "tiny", 12, [4, 24]
        args.prompt_len, args.sweep_scales, args.sweep_k, args.sweep_requests = [8, 48], [0.05], [2], 2
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    cost = GPU_PRICES[args.price]

    env = {
        "python": platform.python_version(), "torch": torch.__version__, "device": args.device,
        "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "preset": args.preset, "params_m": PRESETS[args.preset].num_params / 1e6,
    }
    print(f"# kvforge benchmark\n\n```\n{json.dumps(env)}\n```\n", flush=True)
    report: dict = {"env": env, "args": vars(args)}

    modes = bench_modes(args, cost)
    report["modes"] = modes
    base = modes.get("naive", {}).get("output_tok_s")
    rows = []
    for name, r in modes.items():
        rows.append({"mode": name, **r, "speedup": (r["output_tok_s"] / base) if base else float("nan")})
    print("\n## Throughput / latency\n")
    print(md_table(rows, [
        ("mode", "mode", ""), ("output_tok_s", "out tok/s", ".1f"), ("speedup", "vs naive", ".2f"),
        ("ttft_p50_ms", "TTFT p50 ms", ".0f"), ("ttft_p99_ms", "TTFT p99 ms", ".0f"),
        ("tpot_p50_ms", "TPOT p50 ms", ".1f"), ("e2e_p99_ms", "E2E p99 ms", ".0f"),
        ("mean_batch", "mean batch", ".1f"), ("peak_kv_mib", "peak KV MiB", ".1f"),
        ("weights_mib", "weights MiB", ".1f"), ("usd_per_1m", f"$/1M tok ({cost.name})", ".2f"),
    ]))
    if "speculative" in modes:
        s = modes["speculative"]
        print(f"\nspeculative: accepted/drafted={s['acceptance_rate']:.2f}, "
              f"alpha={s['per_token_acceptance']:.2f}, tokens/step={s['tokens_per_step']:.2f}")

    if not args.no_sweep:
        print("\n## Speculative decoding sweep (batch size 1)\n", flush=True)
        report["spec_sweep"] = sweep = spec_sweep(args, cost)
        print(md_table(sweep, [
            ("residual_scale", "residual scale", ".2f"), ("k", "k", "d"),
            ("acceptance_rate", "accepted/drafted", ".2f"), ("alpha", "α (per-token)", ".2f"),
            ("tokens_per_step", "tok/step", ".2f"), ("theory_tokens_per_step", "theory tok/step", ".2f"),
            ("baseline_tpot_ms", "TPOT base ms", ".1f"), ("spec_tpot_ms", "TPOT spec ms", ".1f"),
            ("speedup", "E2E speedup", ".2f"),
        ]))

    if not args.no_quant_accuracy:
        print("\n## Quantization accuracy\n", flush=True)
        report["quant_accuracy"] = qa = quant_accuracy(args)
        print(md_table(qa, [
            ("mode", "mode", ""), ("weights_mib", "weights MiB", ".1f"), ("ppl", "PPL", ".3f"),
            ("top1_agreement", "top-1 agree", ".3f"), ("kl_divergence", "KL", ".5f"),
        ]))

    print("\n## Roofline projection - Llama-3.1-8B fp16 KV, batch 32, ctx 1024 (ANALYTIC, NOT MEASURED)\n")
    report["roofline"] = rf = roofline()
    print(md_table(rf, [("gpu", "GPU", ""), ("weights", "weights", ""), ("impl", "impl", ""),
                        ("tok_s", "tok/s", ".0f"), ("usd_per_1m", "$/1M tok", ".3f")]))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    tag = args.tag or ("quick" if args.quick else f"{args.preset}-{args.device}")
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out_dir / f'{tag}.json'}")


if __name__ == "__main__":
    main()
