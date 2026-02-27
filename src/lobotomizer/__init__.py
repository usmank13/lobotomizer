"""Lobotomizer — composable model compression for PyTorch."""
from __future__ import annotations

from lobotomizer.core.pipeline import Pipeline
from lobotomizer.stages.base import Stage

__version__ = "0.1.0"
__all__ = ["compress", "Pipeline", "Stage"]


def compress(
    model,
    stages: list[Stage],
    *,
    eval_fn=None,
    calibration_data=None,
    device: str = "cpu",
    constraints: dict | None = None,
):
    """Convenience function: build a Pipeline and run it."""
    return Pipeline(stages).run(
        model,
        eval_fn=eval_fn,
        calibration_data=calibration_data,
        device=device,
        constraints=constraints,
    )
