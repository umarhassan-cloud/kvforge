"""Weight-only INT8, symmetric per-output-channel.

``W ≈ diag(s) · Q`` with ``Q ∈ [-127, 127]``. Because the scale is per *output*
channel it commutes out of the matmul: ``x @ W.T = (x @ Q.T) * s`` — no full-matrix
dequantization is needed; only the int8 → compute-dtype cast of Q.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight [out, in]`` -> ``(qweight int8 [out, in], scale fp32 [out])``."""
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qweight.float() * scale[:, None]


class Int8Linear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        input_scale: torch.Tensor | None = None,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = qweight.shape
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale.to(compute_dtype))
        # AWQ: weight columns were multiplied by s, so the input must be divided by s.
        self.register_buffer(
            "inv_input_scale", None if input_scale is None else (1.0 / input_scale).to(compute_dtype)
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, input_scale: torch.Tensor | None = None) -> Int8Linear:
        if linear.bias is not None:
            raise ValueError("bias-free linears only")
        w = linear.weight.detach().float()
        if input_scale is not None:
            w = w * input_scale[None, :]
        q, s = quantize_int8(w)
        return cls(q, s, input_scale, compute_dtype=linear.weight.dtype)

    def dequantize(self) -> torch.Tensor:
        w = dequantize_int8(self.qweight, self.scale.float())
        if self.inv_input_scale is not None:
            w = w * self.inv_input_scale.float()[None, :]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inv_input_scale is not None:
            x = x * self.inv_input_scale
        return F.linear(x, self.qweight.to(x.dtype)) * self.scale

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bits=8, per_channel"
