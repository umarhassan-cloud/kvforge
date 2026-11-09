"""kvforge: a from-scratch LLM inference engine.

Paged KV cache, continuous batching, speculative decoding and weight-only
INT8/INT4 quantization in readable PyTorch.
"""

from kvforge.batching.request import FinishReason, Request, RequestOutput, RequestStatus
from kvforge.config import (
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
from kvforge.engine import InferenceEngine
from kvforge.model import Transformer
from kvforge.model_loader import PRESETS, ByteTokenizer, build_model, load_hf_model
from kvforge.sampling import SamplingParams

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
