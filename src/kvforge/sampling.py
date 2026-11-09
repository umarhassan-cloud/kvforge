"""Batched sampling with per-request parameters.

:func:`logits_to_probs` returns the *processed* distribution (temperature, top-k,
top-p applied). Speculative decoding's rejection sampler must compare target and
draft under the same processing, so both paths share this function. Greedy rows
(temperature == 0) become one-hot distributions, which makes greedy speculative
decoding fall out of the same rejection rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0  # 0 = disabled
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def logits_to_probs(logits: torch.Tensor, params: Sequence[SamplingParams]) -> torch.Tensor:
    """``logits: [B, V]`` (row ``i`` uses ``params[i]``) -> probabilities ``[B, V]`` in fp32."""
    logits = logits.float()
    vocab = logits.shape[-1]
    device = logits.device
    temps = torch.tensor([p.temperature for p in params], device=device)
    greedy = temps == 0
    scaled = logits / torch.where(greedy, torch.ones_like(temps), temps)[:, None]

    top_ks = torch.tensor([p.top_k if p.top_k > 0 else vocab for p in params], device=device)
    top_ps = torch.tensor([p.top_p for p in params], device=device)
    if bool((top_ks < vocab).any()) or bool((top_ps < 1).any()):
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        rank = torch.arange(vocab, device=device)[None, :]
        remove = rank >= top_ks[:, None]
        sorted_probs = torch.softmax(sorted_logits.masked_fill(remove, float("-inf")), dim=-1)
        # Keep the smallest prefix whose mass reaches top_p (always keep the top token).
        cum_before = sorted_probs.cumsum(-1) - sorted_probs
        remove |= cum_before >= top_ps[:, None]
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    if bool(greedy.any()):
        one_hot = torch.zeros_like(probs).scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        probs = torch.where(greedy[:, None], one_hot, probs)
    return probs


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one token per row of ``[..., V]`` probabilities."""
    flat = probs.reshape(-1, probs.shape[-1])
    out = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return out.view(probs.shape[:-1])


class Sampler:
    def __init__(self, seed: int = 0, device: str | torch.device = "cpu") -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def __call__(
        self, logits: torch.Tensor, params: Sequence[SamplingParams]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(token_ids [B], probs [B, V])``."""
        probs = logits_to_probs(logits, params)
        if all(p.greedy for p in params):
            return probs.argmax(-1), probs
        return sample_from_probs(probs, self.generator), probs
