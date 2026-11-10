"""Model construction: random-init presets for tests/benchmarks and HF checkpoint import.

HF import maps ``LlamaForCausalLM`` / ``GemmaForCausalLM`` / ``MistralForCausalLM`` /
``Qwen2``-style (bias-free) state dicts onto :class:`tokenvault.model.Transformer`.
``transformers`` is an optional dependency, imported lazily.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from tokenvault.config import AttentionBackend, ModelConfig
from tokenvault.model import Transformer

PRESETS: dict[str, ModelConfig] = {
    # Unit tests: runs a full engine step in a few ms on CPU.
    "tiny": ModelConfig(
        vocab_size=512, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
        intermediate_size=172, max_position_embeddings=1024,
    ),
    # CPU benchmarks: large enough that matmuls (not Python) dominate step time.
    "small": ModelConfig(
        vocab_size=8192, hidden_size=512, num_layers=6, num_heads=8, num_kv_heads=4,
        intermediate_size=1408, max_position_embeddings=2048,
    ),
    # Shape-compatible with Llama-3.2-1B (for memory math; random weights).
    "llama-1b": ModelConfig(
        vocab_size=128256, hidden_size=2048, num_layers=16, num_heads=32, num_kv_heads=8,
        intermediate_size=8192, max_position_embeddings=8192, rope_theta=500000.0,
        tie_word_embeddings=True,
    ),
    # Shape-compatible with Llama-3.1-8B.
    "llama-8b": ModelConfig(
        vocab_size=128256, hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
        intermediate_size=14336, max_position_embeddings=8192, rope_theta=500000.0,
    ),
}


def build_model(
    cfg: ModelConfig | str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = 0,
    backend: AttentionBackend = AttentionBackend.GATHER,
    init_std: float = 0.02,
) -> Transformer:
    """Random-init model. Deterministic given ``seed``."""
    if isinstance(cfg, str):
        cfg = PRESETS[cfg]
    if seed is not None:
        torch.manual_seed(seed)
    model = Transformer(cfg, backend)
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=init_std)
        elif "norm" in name:
            torch.nn.init.constant_(p, 1.0 - cfg.norm_weight_offset)
    return model.to(device=device, dtype=dtype).eval()


def config_from_hf(hf_cfg: Any) -> ModelConfig:
    model_type = getattr(hf_cfg, "model_type", "llama")
    is_gemma = model_type.startswith("gemma")
    act = getattr(hf_cfg, "hidden_activation", None) or getattr(hf_cfg, "hidden_act", "silu")
    eos = hf_cfg.eos_token_id[0] if isinstance(hf_cfg.eos_token_id, list) else hf_cfg.eos_token_id
    return ModelConfig(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hf_cfg.hidden_size,
        num_layers=hf_cfg.num_hidden_layers,
        num_heads=hf_cfg.num_attention_heads,
        num_kv_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        intermediate_size=hf_cfg.intermediate_size,
        head_dim=getattr(hf_cfg, "head_dim", None),
        max_position_embeddings=hf_cfg.max_position_embeddings,
        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        rms_norm_eps=hf_cfg.rms_norm_eps,
        hidden_act="gelu_tanh" if is_gemma else act,
        tie_word_embeddings=bool(getattr(hf_cfg, "tie_word_embeddings", is_gemma)),
        scale_embeddings=is_gemma,
        norm_weight_offset=1.0 if is_gemma else 0.0,
        bos_token_id=hf_cfg.bos_token_id,
        eos_token_id=eos,
    )


def load_hf_model(
    name_or_path: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: AttentionBackend = AttentionBackend.GATHER,
) -> Transformer:
    """Load a Llama/Gemma-family HF checkpoint into a tokenvault :class:`Transformer`."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install 'tokenvault[hf]' to load Hugging Face checkpoints") from e

    hf_cfg = AutoConfig.from_pretrained(name_or_path)
    cfg = config_from_hf(hf_cfg)
    hf_model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype)
    state = {k.removeprefix("model."): v for k, v in hf_model.state_dict().items()}
    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["embed_tokens.weight"]
    model = Transformer(cfg, backend)
    result = model.load_state_dict(state, strict=False)
    missing = [m for m in result.missing_keys if not m.endswith("inv_freq")]
    if missing:
        raise RuntimeError(f"checkpoint missing weights: {missing[:8]}")
    del hf_model
    return model.to(device=device, dtype=dtype).eval()


class Tokenizer(Protocol):
    eos_token_id: int | None
    bos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


class ByteTokenizer:
    """Dependency-free UTF-8 byte tokenizer for demos with random-weight models."""

    def __init__(self) -> None:
        self.bos_token_id = 256
        self.eos_token_id = 257
        self.vocab_size = 258

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        return bytes(t for t in token_ids if 0 <= t < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    def __init__(self, name_or_path: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(name_or_path)
        self.eos_token_id = self._tok.eos_token_id
        self.bos_token_id = self._tok.bos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)
