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
    param_count = _count_params(model)
    param_count_trainable = _count_params(model, trainable_only=True)

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


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters including those in quantized modules.

    Quantized modules (e.g. dynamic quantized Linear) store weights in
    packed params rather than exposing them via ``.parameters()``. We
    detect these via ``_weight_bias()`` on leaf modules (modules whose
    parent also has ``_weight_bias`` are skipped to avoid double-counting).
    """
    # First, try the normal .parameters() path
    regular_count = sum(
        p.numel() for p in model.parameters()
        if not trainable_only or p.requires_grad
    )

    if regular_count > 0:
        # Model has normal parameters — no quantized counting needed
        return regular_count

    # Model has no .parameters() — likely fully quantized.
    # Walk leaf modules with _weight_bias() to count packed weights.
    # Only count on modules that DON'T have a child with _weight_bias
    # (avoids double-counting parent + _packed_params child).
    modules_with_wb = {
        id(m) for m in model.modules() if hasattr(m, "_weight_bias")
    }

    quantized_count = 0
    for module in model.modules():
        if not hasattr(module, "_weight_bias"):
            continue
        # Skip if any child also has _weight_bias (we'll count from the child)
        has_child_wb = any(
            id(child) in modules_with_wb
            for child in module.children()
        )
        if has_child_wb:
            continue

        try:
            w, b = module._weight_bias()
            quantized_count += w.numel()
            if b is not None:
                quantized_count += b.numel()
        except (AttributeError, RuntimeError):
            pass

    return quantized_count


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
