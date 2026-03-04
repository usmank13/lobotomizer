"""Stage and adapter registries."""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from lobotomizer.stages.base import Stage

_STAGE_REGISTRY: dict[str, type[Stage]] = {}
_ADAPTER_REGISTRY: dict[str, type[nn.Module]] = {}


def register_stage(name: str) -> Callable:
    """Decorator to register a :class:`Stage` subclass by name."""

    def wrapper(cls: type[Stage]) -> type[Stage]:
        _STAGE_REGISTRY[name] = cls
        return cls

    return wrapper


def register_adapter(name: str) -> Callable:
    """Decorator to register an adapter class by name."""

    def wrapper(cls: type[nn.Module]) -> type[nn.Module]:
        _ADAPTER_REGISTRY[name] = cls
        return cls

    return wrapper


def get_stage(name: str) -> type[Stage]:
    """Look up a registered stage by name."""
    if name not in _STAGE_REGISTRY:
        raise KeyError(f"Unknown stage '{name}'. Registered: {list(_STAGE_REGISTRY)}")
    return _STAGE_REGISTRY[name]


def get_adapter(name: str) -> type[nn.Module]:
    """Look up a registered adapter by name."""
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Registered: {list(_ADAPTER_REGISTRY)}")
    return _ADAPTER_REGISTRY[name]
