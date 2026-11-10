"""Speculative decoding (Leviathan, Kalman & Matias, 2023; Chen et al., 2023).

One speculative step for a batch of decoding sequences:

1. **Draft**: the cheap model proposes ``k`` tokens autoregressively, recording its
   processed distributions ``q_1..q_k``.
2. **Verify**: the target scores ``[x_last, d_1..d_k]`` in ONE forward (query_len = k+1
   through PagedAttention), yielding ``p_1..p_{k+1}``.
3. **Accept/reject** (:func:`rejection_sample`): accept ``d_i`` with probability
   ``min(1, p_i(d_i) / q_i(d_i))``; at the first rejection resample from the residual
   ``norm(max(0, p_i - q_i))``; if all ``k`` are accepted, sample a bonus token from
   ``p_{k+1}``. Every emitted token is distributed *exactly* as the target would have
   sampled it, so quality is unchanged and each step emits between 1 and k+1 tokens.

KV bookkeeping: target K/V for rejected drafts is left in place as stale data — those
slots lie beyond ``num_computed_tokens`` and get overwritten later, so no rollback copy
is needed. The draft shares the target's block tables but owns separate KV tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
from torch import nn

from pulsecache.batching.continuous import BatchItem, ModelRunner
from pulsecache.batching.request import Request
from pulsecache.model import Transformer
from pulsecache.sampling import Sampler, logits_to_probs, sample_from_probs


def rejection_sample(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized speculative rejection sampling.

    Args:
        target_probs: ``[B, k+1, V]`` target distributions at each draft position + bonus.
        draft_probs:  ``[B, k, V]`` draft distributions the drafts were sampled from.
        draft_tokens: ``[B, k]``.

    Returns:
        ``tokens [B, k+1]`` (``-1`` past the emitted prefix) and ``num_accepted [B]``;
        row ``b`` emits ``num_accepted[b] + 1`` tokens.
    """
    b, k = draft_tokens.shape
    device = target_probs.device
    idx = draft_tokens.unsqueeze(-1)
    p_tok = target_probs[:, :k].gather(-1, idx).squeeze(-1)
    q_tok = draft_probs.gather(-1, idx).squeeze(-1)
    accept_prob = (p_tok / q_tok.clamp_min(1e-20)).clamp(max=1.0)
    u = torch.rand((b, k), generator=generator, device=device)
    accepted = u < accept_prob
    num_accepted = accepted.long().cumprod(dim=-1).sum(dim=-1)

    rows = torch.arange(b, device=device)
    p_at = target_probs[rows, num_accepted]
    q_padded = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], dim=1)
    residual = (p_at - q_padded[rows, num_accepted]).clamp_min(0.0)
    mass = residual.sum(-1, keepdim=True)
    # mass == 0 only if p == q at a rejected position (numerically); fall back to p.
    residual = torch.where(mass > 0, residual / mass.clamp_min(1e-20), p_at)
    final = sample_from_probs(residual, generator)

    tokens = torch.cat([draft_tokens, torch.full_like(draft_tokens[:, :1], -1)], dim=1)
    tokens[rows, num_accepted] = final
    keep = torch.arange(k + 1, device=device)[None, :] <= num_accepted[:, None]
    return tokens.masked_fill(~keep, -1), num_accepted


@dataclass
class SpecDecodeStats:
    num_steps: int = 0
    num_seq_steps: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_emitted_tokens: int = 0
    num_rejections: int = 0
    accepted_length_hist: dict[int, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted."""
        return self.num_accepted_tokens / self.num_draft_tokens if self.num_draft_tokens else 0.0

    @property
    def per_token_acceptance(self) -> float:
        """MLE of the per-position acceptance probability α used by the Leviathan speedup
        formula. Drafts after the first rejection are never tested, so
        ``acceptance_rate`` underestimates α; here each step contributes its accepted
        tokens as successes and at most one rejection as a failure."""
        trials = self.num_accepted_tokens + self.num_rejections
        return self.num_accepted_tokens / trials if trials else 0.0

    @property
    def mean_tokens_per_step(self) -> float:
        """Average tokens emitted per sequence per target forward (1.0 == no speedup)."""
        return self.num_emitted_tokens / self.num_seq_steps if self.num_seq_steps else 0.0

    def as_dict(self) -> dict[str, float | int | dict[int, int]]:
        return {
            "acceptance_rate": round(self.acceptance_rate, 4),
            "per_token_acceptance": round(self.per_token_acceptance, 4),
            "mean_tokens_per_step": round(self.mean_tokens_per_step, 4),
            "num_draft_tokens": self.num_draft_tokens,
            "num_accepted_tokens": self.num_accepted_tokens,
            "accepted_length_hist": dict(sorted(self.accepted_length_hist.items())),
        }


@dataclass
class SpecStepResult:
    new_tokens: list[list[int]]
    num_accepted: list[int]
    draft_computed: list[int]


class SpeculativeDecoder:
    def __init__(
        self,
        draft_runner: ModelRunner,
        target_runner: ModelRunner,
        sampler: Sampler,
        num_speculative_tokens: int,
    ) -> None:
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        self.draft = draft_runner
        self.target = target_runner
        self.sampler = sampler
        self.k = num_speculative_tokens
        self.stats = SpecDecodeStats()

    def step(self, requests: Sequence[Request], k: int | None = None) -> SpecStepResult:
        """Propose ``k`` tokens per request and verify. Requests must be in decode phase
        with KV slots reserved for ``num_tokens + k`` positions."""
        k = self.k if k is None else k
        params = [r.sampling_params for r in requests]
        n = [r.num_tokens for r in requests]
        gen = self.sampler.generator

        # 1) Draft. The first step also catches the draft cache up (1-2 tokens normally).
        items = [
            BatchItem(r.request_id, r.all_token_ids[r.num_draft_computed_tokens :],
                      r.num_draft_computed_tokens, "last")
            for r in requests
        ]
        drafts: list[torch.Tensor] = []
        draft_probs: list[torch.Tensor] = []
        for j in range(k):
            q = logits_to_probs(self.draft.execute(items), params)
            d = sample_from_probs(q, gen)
            drafts.append(d)
            draft_probs.append(q)
            if j + 1 < k:
                toks = d.tolist()
                items = [
                    BatchItem(r.request_id, [toks[i]], n[i] + j, "last")
                    for i, r in enumerate(requests)
                ]
        draft_tokens = torch.stack(drafts, dim=1)  # [B, k]
        q_all = torch.stack(draft_probs, dim=1)  # [B, k, V]

        # 2) Verify all k+1 positions in one target forward.
        dt = draft_tokens.tolist()
        verify = [
            BatchItem(r.request_id, [r.all_token_ids[-1], *dt[i]], n[i] - 1, "all")
            for i, r in enumerate(requests)
        ]
        logits = self.target.execute(verify)
        vocab = logits.shape[-1]
        rep_params = [p for p in params for _ in range(k + 1)]
        p_all = logits_to_probs(logits, rep_params).view(len(requests), k + 1, vocab)

        # 3) Accept / reject.
        tokens, num_acc = rejection_sample(p_all, q_all.to(p_all.device), draft_tokens, gen)
        acc = num_acc.tolist()
        new_tokens = [[t for t in row if t >= 0] for row in tokens.tolist()]
        # Draft K/V is valid through position n+k-2 and only where tokens were accepted.
        draft_computed = [min(n[i] + k - 1, n[i] + acc[i]) for i in range(len(requests))]

        s = self.stats
        s.num_steps += 1
        s.num_seq_steps += len(requests)
        s.num_draft_tokens += k * len(requests)
        s.num_accepted_tokens += sum(acc)
        s.num_emitted_tokens += sum(len(t) for t in new_tokens)
        s.num_rejections += sum(1 for a in acc if a < k)
        for a in acc:
            s.accepted_length_hist[a] = s.accepted_length_hist.get(a, 0) + 1
        return SpecStepResult(new_tokens, acc, draft_computed)


def make_layer_skip_draft(target: Transformer, num_layers: int) -> Transformer:
    """Self-speculative draft: the first ``num_layers`` decoder layers of ``target`` plus
    its embeddings, final norm and LM head (parameters shared, zero extra weight memory)."""
    if not 0 < num_layers < target.num_layers:
        raise ValueError(f"draft layers must be in [1, {target.num_layers - 1}]")
    draft = Transformer.__new__(Transformer)
    nn.Module.__init__(draft)
    draft.config = replace(target.config, num_layers=num_layers)
    draft.embed_tokens = target.embed_tokens
    draft.layers = nn.ModuleList(list(target.layers[:num_layers]))
    draft.norm = target.norm
    draft.lm_head = target.lm_head
    draft.rotary = target.rotary
    return draft.eval()
