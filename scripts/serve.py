"""Launch the OpenAI-compatible kvforge server.

Examples::

    # Random-weight demo model + byte tokenizer, no downloads:
    python scripts/serve.py --preset tiny

    # Real checkpoint (needs `pip install -e .[hf]`), INT8 weights, speculative decoding:
    python scripts/serve.py --hf-model meta-llama/Llama-3.2-1B --quant int8 --speculative 4 \
        --device cuda --dtype bfloat16 --kv-cache-gb 8
"""

from __future__ import annotations

import argparse

import torch

from kvforge import (
    PRESETS,
    AttentionBackend,
    ByteTokenizer,
    CacheConfig,
    EngineConfig,
    InferenceEngine,
    QuantConfig,
    QuantMode,
    SchedulerConfig,
    SpeculativeConfig,
    build_model,
    load_hf_model,
)
from kvforge.metrics import GPU_PRICES
from kvforge.model_loader import HFTokenizer
from kvforge.serving import create_app

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    src.add_argument("--hf-model", help="HF hub id or local path (Llama/Gemma/Mistral family)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--kv-cache-gb", type=float, default=0.25)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--prefix-caching", action="store_true")
    p.add_argument("--speculative", type=int, default=0, metavar="K", help="draft tokens per step")
    p.add_argument("--draft-layers", type=int, default=1)
    p.add_argument("--quant", choices=[m.value for m in QuantMode], default="none")
    p.add_argument("--attention", choices=[b.value for b in AttentionBackend], default="gather")
    p.add_argument("--gpu-price", choices=sorted(GPU_PRICES), default="cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    import uvicorn

    dtype = DTYPES[args.dtype]
    if args.hf_model:
        model = load_hf_model(args.hf_model, args.device, dtype)
        tokenizer = HFTokenizer(args.hf_model)
        name = args.hf_model
    else:
        model = build_model(args.preset, args.device, dtype)
        tokenizer = ByteTokenizer()
        name = f"kvforge-{args.preset}-random"

    config = EngineConfig(
        model=model.config,
        cache=CacheConfig(
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            kv_cache_memory_gb=None if args.num_blocks else args.kv_cache_gb,
            enable_prefix_caching=args.prefix_caching,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_model_len=args.max_model_len,
        ),
        speculative=SpeculativeConfig(args.speculative, args.draft_layers) if args.speculative else None,
        quant=QuantConfig(mode=QuantMode(args.quant)),
        attention_backend=AttentionBackend(args.attention),
        device=args.device,
        dtype=dtype,
    )
    engine = InferenceEngine(config, model=model, tokenizer=tokenizer)
    kv = engine.kv_cache
    print(
        f"kvforge: model={name} blocks={kv.num_blocks}x{kv.block_size} "
        f"kv={kv.memory_bytes / 2**20:.1f} MiB device={args.device}"
    )
    uvicorn.run(create_app(engine, name, GPU_PRICES[args.gpu_price]), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
