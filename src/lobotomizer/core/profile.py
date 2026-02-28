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
    """Count parameters, handling both regular and quantized modules.

    torchao-quantized modules still expose ``.parameters()`` but they
    contain packed int8 weights + scale/zero-point tensors, which inflates
    naive counts.  We walk leaf Linear modules and use their
    ``in_features``/``out_features`` to compute the *effective* param count
    for quantized layers, falling back to ``.numel()`` for regular params.
    """
    # Collect IDs of parameters owned by quantized Linear modules so we
    # can skip them in the regular parameter walk.
    quantized_param_ids: set[int] = set()
    effective_quantized_count = 0

    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        # Detect torchao quantization: weight won't be a plain Tensor
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        # Detect torchao quantization: check the weight (or its .data)
        # for torchao types, which aren't plain nn.Parameter/Tensor
        w_to_check = weight.data if isinstance(weight, nn.Parameter) else weight
        w_module = type(w_to_check).__module__ or ""
        is_quantized = "torchao" in w_module
        if not is_quantized:
            # Fallback: non-float dtype on a plain parameter
            try:
                is_quantized = not weight.dtype.is_floating_point
            except Exception:
                pass

        if is_quantized:
            # Use the module's shape attributes for effective count
            effective_quantized_count += module.in_features * module.out_features
            if module.bias is not None:
                effective_quantized_count += module.out_features
            # Mark all parameters of this module to skip later
            for p in module.parameters():
                quantized_param_ids.add(id(p))

    # Count remaining (non-quantized) parameters normally
    regular_count = sum(
        p.numel() for p in model.parameters()
        if id(p) not in quantized_param_ids
        and (not trainable_only or p.requires_grad)
    )

    # For trainable_only, quantized params are frozen so return 0 for them
    if trainable_only:
        return regular_count

    return regular_count + effective_quantized_count


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
