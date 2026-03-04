"""Forward hook manager for capturing intermediate activations."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


class ForwardHookManager:
    """Register forward hooks on named layers and collect their outputs."""

    def __init__(self) -> None:
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[RemovableHandle] = []

    def _make_hook(self, name: str):
        """Create a hook closure that stores the layer output."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Handle modules that return tuples (e.g., transformers)
            if isinstance(output, tuple):
                self._features[name] = output[0]
            else:
                self._features[name] = output

        return hook

    def register(self, model: nn.Module, layer_names: list[str]) -> None:
        """Attach forward hooks to the specified named modules."""
        modules = dict(model.named_modules())
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"Module '{name}' not found in model")
            hook = modules[name].register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)
            logger.debug("Registered hook on '%s'", name)

    def pop_features(self) -> dict[str, torch.Tensor]:
        """Return captured features and clear the buffer."""
        feats = dict(self._features)
        self._features.clear()
        return feats

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
