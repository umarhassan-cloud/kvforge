"""Typed configuration objects for every kvforge subsystem.

All configs are plain dataclasses so they serialize trivially (``dataclasses.asdict``)
into benchmark reports and can be constructed from CLI flags without a framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import torch


class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class QuantMode(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"


class AttentionBackend(str, Enum):
    # Materialize the gathered KV for the whole context, then one softmax.
    GATHER = "gather"
    # Stream over KV blocks with an online softmax (flash-decoding style).
    ONLINE = "online"


@dataclass
class ModelConfig:
    """Decoder-only transformer hyper-parameters (Llama / Gemma family)."""

    vocab_size: int = 512
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 344
    head_dim: int | None = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"  # "silu" (Llama SwiGLU) or "gelu_tanh" (Gemma GeGLU)
    tie_word_embeddings: bool = False
    # Gemma-specific quirks.
    scale_embeddings: bool = False  # multiply embeddings by sqrt(hidden_size)
    norm_weight_offset: float = 0.0  # Gemma RMSNorm uses (1 + w)
    bos_token_id: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.num_heads != 0:
                raise ValueError("hidden_size must be divisible by num_heads when head_dim is unset")
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be a multiple of num_kv_heads (GQA)")

    @property
    def num_params(self) -> int:
        """Approximate parameter count (excluding norms)."""
        assert self.head_dim is not None
        h, d = self.hidden_size, self.head_dim
        attn = h * d * (self.num_heads * 2 + self.num_kv_heads * 2)
        mlp = 3 * h * self.intermediate_size
        emb = self.vocab_size * h * (1 if self.tie_word_embeddings else 2)
        return self.num_layers * (attn + mlp) + emb


@dataclass
class CacheConfig:
    """Paged KV-cache configuration.

    Either ``num_blocks`` is given explicitly, or it is derived from
    ``kv_cache_memory_gb`` and the model shape in :meth:`resolve_num_blocks`.
    """

    block_size: int = 16
    num_blocks: int | None = 256
    kv_cache_memory_gb: float | None = None
    enable_prefix_caching: bool = False
    # Fraction of blocks kept free at admission time to avoid immediate preemption.
    watermark: float = 0.01
    dtype: torch.dtype = torch.float32

    def bytes_per_block(self, model: ModelConfig) -> int:
        assert model.head_dim is not None
        elem = torch.tensor([], dtype=self.dtype).element_size()
        # K and V, every layer.
        return 2 * model.num_layers * self.block_size * model.num_kv_heads * model.head_dim * elem

    def resolve_num_blocks(self, model: ModelConfig) -> int:
        if self.num_blocks is not None:
            return self.num_blocks
        if self.kv_cache_memory_gb is None:
            raise ValueError("either num_blocks or kv_cache_memory_gb must be set")
        return max(1, math.floor(self.kv_cache_memory_gb * 1024**3 / self.bytes_per_block(model)))


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    max_model_len: int = 2048
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    enable_chunked_prefill: bool = True


@dataclass
class SpeculativeConfig:
    """Draft-model speculative decoding (Leviathan et al., 2023)."""

    num_speculative_tokens: int = 4
    # When no explicit draft model is passed to the engine, build a layer-skip
    # self-draft that shares embeddings / lm_head with the target.
    draft_num_layers: int | None = 1


@dataclass
class QuantConfig:
    mode: QuantMode = QuantMode.NONE
    group_size: int = 128  # INT4 group size along the input dimension
    quantize_lm_head: bool = False
    # Activation-aware per-channel scaling search (AWQ-style) when calibration data is given.
    awq: bool = True
    awq_grid: int = 20


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    speculative: SpeculativeConfig | None = None
    quant: QuantConfig = field(default_factory=QuantConfig)
    attention_backend: AttentionBackend = AttentionBackend.GATHER
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0

    def __post_init__(self) -> None:
        self.scheduler.max_model_len = min(
            self.scheduler.max_model_len, self.model.max_position_embeddings
        )
        if self.cache.dtype != self.dtype:
            self.cache.dtype = self.dtype
