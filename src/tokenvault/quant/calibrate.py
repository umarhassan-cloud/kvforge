"""Calibration, model conversion and accuracy evaluation for weight-only quantization.

* :func:`collect_linear_inputs` — forward hooks capture a row-subsample of each
  projection's input activations on calibration sequences.
* :func:`awq_search_scale` — activation-aware scaling (Lin et al., 2023, "AWQ"): salient
  input channels (large mean |x|) are scaled up by ``s = mean|x|^α`` before rounding and
  the input is divided by ``s``; α is grid-searched to minimize output MSE. α=0 is plain
  round-to-nearest, so the search can never be worse than RTN on calibration data.
* :func:`quantize_model` — swaps ``nn.Linear`` modules for INT8 / INT4 modules in place.
* :func:`evaluate_perplexity`, :func:`logit_fidelity` — accuracy side of the trade-off.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import nn

from tokenvault.config import QuantConfig, QuantMode
from tokenvault.model import Transformer
from tokenvault.quant.int4 import Int4Linear, dequantize_int4, quantize_int4
from tokenvault.quant.int8 import Int8Linear, dequantize_int8, quantize_int8

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _target_linears(model: Transformer, include_lm_head: bool) -> dict[str, nn.Linear]:
    out = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in PROJECTIONS
    }
    if include_lm_head and not model.config.tie_word_embeddings:
        out["lm_head"] = model.lm_head
    return out


@torch.inference_mode()
def collect_linear_inputs(
    model: Transformer,
    calib_batches: Iterable[torch.Tensor],
    names: Iterable[str],
    max_rows: int = 512,
) -> dict[str, torch.Tensor]:
    """Return ``{linear_name: [rows, in_features]}`` sampled input activations."""
    modules = dict(model.named_modules())
    store: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    handles = []
    for name in store:
        def hook(_mod: nn.Module, args: tuple[torch.Tensor, ...], _out: torch.Tensor, n: str = name) -> None:
            store[n].append(args[0].detach().reshape(-1, args[0].shape[-1]).float().cpu())
        handles.append(modules[name].register_forward_hook(hook))
    try:
        for batch in calib_batches:
            model.forward_dense(batch.to(next(model.parameters()).device))
    finally:
        for h in handles:
            h.remove()
    gen = torch.Generator().manual_seed(0)
    result = {}
    for n, chunks in store.items():
        x = torch.cat(chunks)
        if x.shape[0] > max_rows:
            x = x[torch.randperm(x.shape[0], generator=gen)[:max_rows]]
        result[n] = x
    return result


def _fake_quant(mode: QuantMode, group_size: int) -> Callable[[torch.Tensor], torch.Tensor]:
    if mode == QuantMode.INT8:
        return lambda w: dequantize_int8(*quantize_int8(w))

    def int4(w: torch.Tensor) -> torch.Tensor:
        packed, s, z, g = quantize_int4(w, group_size)
        return dequantize_int4(packed, s, z, g)

    return int4


@torch.inference_mode()
def awq_search_scale(
    weight: torch.Tensor,
    x: torch.Tensor,
    fake_quant: Callable[[torch.Tensor], torch.Tensor],
    grid: int = 20,
) -> tuple[torch.Tensor | None, float, float]:
    """Grid-search ``s = mean|x|^α``. Returns ``(best_scale or None for RTN, rtn_mse, best_mse)``."""
    w = weight.detach().float().cpu()
    x = x.float()
    ref = x @ w.T
    act = x.abs().mean(0).clamp_min(1e-6)
    rtn_mse = float((x @ fake_quant(w).T - ref).pow(2).mean())
    best_scale, best_mse = None, rtn_mse
    for i in range(1, grid):
        alpha = i / grid
        s = act.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()  # keep the dynamic range centred around 1
        mse = float(((x / s) @ fake_quant(w * s).T - ref).pow(2).mean())
        if mse < best_mse:
            best_scale, best_mse = s, mse
    return best_scale, rtn_mse, best_mse


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, attr = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, attr, new)


def quantize_model(
    model: Transformer,
    config: QuantConfig,
    calib_batches: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Quantize projection weights in place. Returns a small report."""
    mode = QuantMode(config.mode)
    if mode == QuantMode.NONE:
        return {}
    bytes_before = weight_memory_bytes(model)
    targets = _target_linears(model, config.quantize_lm_head)
    acts = (
        collect_linear_inputs(model, calib_batches, targets)
        if (calib_batches and config.awq)
        else {}
    )
    fq = _fake_quant(mode, config.group_size)
    rtn_err = awq_err = 0.0
    for name, linear in targets.items():
        scale = None
        if name in acts:
            scale, rtn, best = awq_search_scale(linear.weight, acts[name], fq, config.awq_grid)
            rtn_err += rtn
            awq_err += best
            if scale is not None:
                scale = scale.to(linear.weight.device)
        if mode == QuantMode.INT8:
            new: nn.Module = Int8Linear.from_linear(linear, input_scale=scale)
        else:
            new = Int4Linear.from_linear(linear, config.group_size, input_scale=scale)
        _set_module(model, name, new.to(linear.weight.device))
    report = {
        "num_quantized_linears": float(len(targets)),
        "weight_bytes_before": float(bytes_before),
        "weight_bytes_after": float(weight_memory_bytes(model)),
    }
    if acts:
        report["calib_rtn_mse"] = rtn_err
        report["calib_awq_mse"] = awq_err
    return report


def weight_memory_bytes(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    tensors = [p for p in model.parameters()] + [
        b for n, b in model.named_buffers() if not n.endswith("inv_freq")
    ]
    for t in tensors:
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


@torch.inference_mode()
def evaluate_perplexity(model: Transformer, sequences: torch.Tensor, batch_size: int = 8) -> float:
    """Token-level perplexity of ``sequences [N, T]`` (teacher forcing)."""
    device = next(model.parameters()).device
    nll, count = 0.0, 0
    for i in range(0, sequences.shape[0], batch_size):
        batch = sequences[i : i + batch_size].to(device)
        logits, _ = model.forward_dense(batch[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
        )
        nll += float(loss)
        count += batch[:, 1:].numel()
    return math.exp(nll / count)


@torch.inference_mode()
def logit_fidelity(
    reference: Transformer, candidate: Transformer, sequences: torch.Tensor
) -> dict[str, float]:
    """Top-1 agreement and mean KL(ref || cand) of next-token distributions."""
    device = next(reference.parameters()).device
    seq = sequences.to(device)
    ref_logits, _ = reference.forward_dense(seq)
    cand_logits, _ = candidate.forward_dense(seq)
    ref_lp = torch.log_softmax(ref_logits, -1)
    cand_lp = torch.log_softmax(cand_logits, -1)
    kl = (ref_lp.exp() * (ref_lp - cand_lp)).sum(-1).mean()
    agree = (ref_logits.argmax(-1) == cand_logits.argmax(-1)).float().mean()
    return {"top1_agreement": float(agree), "kl_divergence": float(kl)}
