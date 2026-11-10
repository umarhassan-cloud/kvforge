"""tokenvault: a from-scratch LLM inference engine.

Paged KV cache, continuous batching, speculative decoding and weight-only
INT8/INT4 quantization in readable PyTorch.
"""

from tokenvault.batching.request import FinishReason, Request, RequestOutput, RequestStatus
from tokenvault.config import (
    AttentionBackend,
    CacheConfig,
    EngineConfig,
    ModelConfig,
    QuantConfig,
    QuantMode,
    SchedulerConfig,
    SchedulingPolicy,
    SpeculativeConfig,
)
from tokenvault.engine import InferenceEngine
from tokenvault.model import Transformer
from tokenvault.model_loader import PRESETS, ByteTokenizer, build_model, load_hf_model
from tokenvault.sampling import SamplingParams

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "AttentionBackend",
    "ByteTokenizer",
    "CacheConfig",
    "EngineConfig",
    "FinishReason",
    "InferenceEngine",
    "ModelConfig",
    "QuantConfig",
    "QuantMode",
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    "SchedulerConfig",
    "SchedulingPolicy",
    "SpeculativeConfig",
    "Transformer",
    "__version__",
    "build_model",
    "load_hf_model",
]
