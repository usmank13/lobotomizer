"""Feature adapters for dimension alignment in knowledge distillation."""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn

from lobotomizer.core.registry import _ADAPTER_REGISTRY, register_adapter  # noqa: F401

logger = logging.getLogger(__name__)


class FeatureAdapter(nn.Module):
    """Linear projection adapter for dimension mismatch.

    Uses ``nn.Identity`` when dimensions already match, otherwise
    a learned ``nn.Linear`` projection.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        if student_dim == teacher_dim:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FunctionalAdapter(nn.Module):
    """Wraps an arbitrary callable as an ``nn.Module`` adapter."""

    def __init__(self, fn: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def resolve_adapter(
    adapter: Any,
    student_dim: int,
    teacher_dim: int,
) -> nn.Module:
    """Resolve an adapter specification into an ``nn.Module``.

    Parameters
    ----------
    adapter
        One of: ``None`` (auto), ``str`` (registry lookup), a ``type``
        (instantiated), an ``nn.Module`` (used directly), or a callable
        (wrapped in :class:`FunctionalAdapter`).
    student_dim : int
        Dimensionality of the student feature.
    teacher_dim : int
        Dimensionality of the teacher feature.
    """
    if adapter is None:
        return FeatureAdapter(student_dim, teacher_dim)
    if isinstance(adapter, str):
        if adapter not in _ADAPTER_REGISTRY:
            raise KeyError(
                f"Unknown adapter '{adapter}'. Registered: {list(_ADAPTER_REGISTRY)}"
            )
        return _ADAPTER_REGISTRY[adapter](student_dim, teacher_dim)
    if isinstance(adapter, type):
        return adapter(student_dim, teacher_dim)
    if isinstance(adapter, nn.Module):
        return adapter
    if callable(adapter):
        return FunctionalAdapter(adapter)
    raise TypeError(f"Cannot resolve adapter of type {type(adapter)}")
