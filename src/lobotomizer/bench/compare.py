"""Before/after model comparison."""
from __future__ import annotations

from typing import Any

import torch.nn as nn

from lobotomizer.core.profile import profile_model


def compare(
    before: nn.Module,
    after: nn.Module,
    input_shape: tuple | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Return a dict with before, after, and delta profiles."""
    p_before = profile_model(before, input_shape=input_shape, device=device)
    p_after = profile_model(after, input_shape=input_shape, device=device)
    delta: dict[str, Any] = {}
    for k in p_before:
        b, a = p_before[k], p_after[k]
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta[k] = a - b
        else:
            delta[k] = {"before": b, "after": a}
    return {"before": p_before, "after": p_after, "delta": delta}
