"""Abstract Stage interface and shared data structures."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages."""

    original_model: nn.Module
    eval_fn: Callable | None = None
    target_constraints: dict = field(default_factory=dict)
    history: list[StageResult] = field(default_factory=list)
    calibration_data: DataLoader | None = None
    device: str = "cpu"


@dataclass
class StageResult:
    """Result from a single stage execution."""

    stage_name: str
    model: nn.Module
    metrics_before: dict
    metrics_after: dict
    metadata: dict = field(default_factory=dict)

    @property
    def delta(self) -> dict:
        """Compute deltas between before and after metrics."""
        all_keys = set(self.metrics_before) | set(self.metrics_after)
        result: dict[str, Any] = {}
        for k in sorted(all_keys):
            before = self.metrics_before.get(k)
            after = self.metrics_after.get(k)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                result[k] = after - before
            else:
                result[k] = {"before": before, "after": after}
        return result


class Stage(ABC):
    """Base class for all compression stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        """Apply this compression stage. Returns modified model."""
        ...

    def validate(self, model: nn.Module) -> list[str]:
        """Check compatibility. Return list of warnings/errors."""
        return []

    def estimate_impact(self, model: nn.Module) -> dict:
        """Estimate compression impact without applying."""
        return {}
