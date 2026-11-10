"""Weight-only INT4, asymmetric group-wise (GPTQ/AWQ storage layout).

Each row is split into groups of ``group_size`` input channels; each group stores a
scale and a zero-point so ``w ≈ (q - z) * s`` with ``q ∈ [0, 15]``. Two 4-bit codes are
packed per byte (even column in the low nibble), giving 0.5 B/weight plus
``2 * 4 B / group_size`` of metadata (≈0.56 B/weight at g=128 vs 2 B for fp16).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in ``[0, 15]`` of shape ``[..., n]`` (n even) into ``[..., n // 2]``."""
    if q.shape[-1] % 2:
        raise ValueError("last dim must be even to pack int4")
    q = q.to(torch.uint8)
    return q[..., 0::2] | (q[..., 1::2] << 4)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = packed >> 4
    return torch.stack([low, high], dim=-1).flatten(-2)


def resolve_group_size(in_features: int, group_size: int) -> int:
    """Largest even divisor of ``in_features`` that is <= ``group_size``."""
    g = min(group_size, in_features)
    while g > 2 and (in_features % g or g % 2):
        g -= 1
    return g


def quantize_int4(
    weight: torch.Tensor, group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """``weight [out, in]`` -> ``(packed uint8 [out, in/2], scales [out, G], zeros [out, G], g)``."""
    out_f, in_f = weight.shape
    g = resolve_group_size(in_f, group_size)
    w = weight.detach().float().view(out_f, in_f // g, g)
    w_min = w.amin(dim=-1, keepdim=True).clamp(max=0.0)
    w_max = w.amax(dim=-1, keepdim=True).clamp(min=0.0)
    scale = ((w_max - w_min) / 15.0).clamp_min(1e-8)
    zero = torch.round(-w_min / scale).clamp(0, 15)
    q = (torch.round(w / scale) + zero).clamp(0, 15).to(torch.uint8)
    return pack_int4(q.view(out_f, in_f)), scale.squeeze(-1), zero.squeeze(-1), g


def dequantize_int4(
    packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_f = packed.shape[0]
    q = unpack_int4(packed).float().view(out_f, -1, group_size)
    return ((q - zeros.float()[..., None]) * scales.float()[..., None]).view(out_f, -1)


class Int4Linear(nn.Module):
    def __init__(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        group_size: int,
        input_scale: torch.Tensor | None = None,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.out_features = packed.shape[0]
        self.in_features = packed.shape[1] * 2
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.register_buffer("qweight", packed)
        self.register_buffer("scales", scales.to(compute_dtype))
        self.register_buffer("zeros", zeros.to(torch.uint8))
        self.register_buffer(
            "inv_input_scale", None if input_scale is None else (1.0 / input_scale).to(compute_dtype)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 128, input_scale: torch.Tensor | None = None
    ) -> Int4Linear:
        if linear.bias is not None:
            raise ValueError("bias-free linears only")
        w = linear.weight.detach().float()
        if input_scale is not None:
            w = w * input_scale[None, :]
        packed, s, z, g = quantize_int4(w, group_size)
        return cls(packed, s, z, g, input_scale, compute_dtype=linear.weight.dtype)

    def dequantize(self) -> torch.Tensor:
        w = dequantize_int4(self.qweight, self.scales, self.zeros, self.group_size)
        if self.inv_input_scale is not None:
            w = w * self.inv_input_scale.float()[None, :]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_input_scale is not None:
            x = x * self.inv_input_scale
        # Dequantize-then-GEMM reference path. A fused kernel (Marlin / tinygemm) unpacks in
        # registers and never materializes W; here W lives only for the duration of the call.
        w = dequantize_int4(self.qweight, self.scales, self.zeros, self.group_size)
        return F.linear(x, w.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits=4, group_size={self.group_size}"
