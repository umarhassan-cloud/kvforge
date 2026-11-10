"""pulsecache: a from-scratch LLM inference engine.

Paged KV cache, continuous batching, speculative decoding and weight-only
INT8/INT4 quantization in readable PyTorch.
"""

from pulsecache.batching.request import FinishReason, Request, RequestOutput, RequestStatus
from pulsecache.config import (
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
from pulsecache.engine import InferenceEngine
from pulsecache.model import Transformer
from pulsecache.model_loader import PRESETS, ByteTokenizer, build_model, load_hf_model
from pulsecache.sampling import SamplingParams

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
