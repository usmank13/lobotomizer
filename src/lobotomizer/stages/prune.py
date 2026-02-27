"""Pruning stage using PyTorch's built-in torch.nn.utils.prune."""
from __future__ import annotations

import logging
from typing import Sequence

import torch.nn as nn
import torch.nn.utils.prune as prune

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)

_DEFAULT_TARGETS = (nn.Linear, nn.Conv2d)

_METHODS = {
    "l1_unstructured": "l1_unstructured",
    "random_unstructured": "random_unstructured",
    "l1_structured": "l1_structured",
    "random_structured": "random_structured",
}


class Prune(Stage):
    """Prune model weights using PyTorch native pruning utilities."""

    def __init__(
        self,
        method: str = "l1_unstructured",
        sparsity: float = 0.3,
        target_modules: Sequence[type] | None = None,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"Unknown pruning method '{method}'. Choose from: {list(_METHODS)}")
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"Sparsity must be in [0, 1], got {sparsity}")
        self._method = method
        self._sparsity = sparsity
        self._target_modules = tuple(target_modules) if target_modules else _DEFAULT_TARGETS

    @property
    def name(self) -> str:
        return f"prune({self._method}, {self._sparsity})"

    def _get_prunable_layers(self, model: nn.Module) -> list[tuple[nn.Module, str]]:
        """Return (module, param_name) pairs to prune."""
        layers: list[tuple[nn.Module, str]] = []
        for module in model.modules():
            if isinstance(module, self._target_modules):
                layers.append((module, "weight"))
        return layers

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        layers = self._get_prunable_layers(model)
        if not layers:
            warnings.append("Model has no prunable layers matching target module types.")
        if self._method in ("l1_structured", "random_structured"):
            has_conv = any(isinstance(m, nn.Conv2d) for m, _ in layers)
            if not has_conv:
                warnings.append("Structured pruning selected but no Conv2d layers found.")
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        layers = self._get_prunable_layers(model)
        total_params = sum(p.numel() for p in model.parameters())
        prunable_params = sum(getattr(m, name).numel() for m, name in layers)
        estimated_removed = int(prunable_params * self._sparsity)
        return {
            "total_params": total_params,
            "prunable_params": prunable_params,
            "estimated_removed": estimated_removed,
            "estimated_remaining": total_params - estimated_removed,
        }

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        layers = self._get_prunable_layers(model)
        if not layers:
            logger.warning("No prunable layers found; returning model unchanged.")
            return model

        for module, param_name in layers:
            if self._method == "l1_unstructured":
                prune.l1_unstructured(module, param_name, amount=self._sparsity)
            elif self._method == "random_unstructured":
                prune.random_unstructured(module, param_name, amount=self._sparsity)
            elif self._method == "l1_structured":
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, param_name, amount=self._sparsity, n=1, dim=0)
                else:
                    prune.l1_unstructured(module, param_name, amount=self._sparsity)
            elif self._method == "random_structured":
                if isinstance(module, nn.Conv2d):
                    prune.random_structured(module, param_name, amount=self._sparsity, dim=0)
                else:
                    prune.random_unstructured(module, param_name, amount=self._sparsity)

        # Make pruning permanent
        for module, param_name in layers:
            prune.remove(module, param_name)

        return model
