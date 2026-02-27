"""Model profiling utilities."""
from __future__ import annotations

import io
from typing import Any

import torch
import torch.nn as nn


def profile_model(
    model: nn.Module,
    input_shape: tuple | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Profile a model and return key metrics.

    Returns a dict with:
    - param_count: total number of parameters
    - param_count_trainable: trainable parameters
    - size_mb: approximate model size in MB (state_dict serialized)
    - flops: estimated FLOPs (only if *input_shape* provided and a profiling
      backend is available; ``None`` otherwise)
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Size via serialized state_dict
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    size_mb = buf.tell() / (1024 * 1024)

    flops: int | None = None
    if input_shape is not None:
        flops = _estimate_flops(model, input_shape, device)

    return {
        "param_count": param_count,
        "param_count_trainable": param_count_trainable,
        "size_mb": round(size_mb, 4),
        "flops": flops,
    }


def _estimate_flops(
    model: nn.Module,
    input_shape: tuple,
    device: str,
) -> int | None:
    """Try to estimate FLOPs using available backends."""
    dummy = torch.randn(*input_shape, device=device)

    # Try torchprofile
    try:
        from torchprofile import profile_macs  # type: ignore[import-untyped]

        return int(profile_macs(model.to(device), (dummy,))) * 2
    except ImportError:
        pass

    # Try ptflops
    try:
        from ptflops import get_model_complexity_info  # type: ignore[import-untyped]

        macs, _ = get_model_complexity_info(
            model,
            input_shape[1:],
            as_strings=False,
            print_per_layer_stat=False,
        )
        return int(macs) * 2
    except ImportError:
        pass

    return None
