"""Lobotomizer — composable model compression for PyTorch."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from lobotomizer.core.pipeline import Pipeline
from lobotomizer.core.recipe import build_pipeline_from_recipe
from lobotomizer.core.result import Result
from lobotomizer.stages.base import Stage
from lobotomizer.stages.prune import Prune
from lobotomizer.stages.quantize import Quantize
from lobotomizer.stages.structured_prune import StructuredPrune

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

__version__ = "0.1.0"
__all__ = ["Pipeline", "Prune", "Quantize", "Result", "Stage", "StructuredPrune", "compress"]


def compress(
    model: nn.Module,
    recipe: str | list[Stage] = "balanced",
    *,
    eval_fn: Callable | None = None,
    calibration_data: DataLoader | None = None,
    device: str = "cpu",
    constraints: dict | None = None,
    input_shape: tuple | None = None,
    **kwargs,
) -> Result:
    """One-liner compression using a named recipe or list of stages.

    Parameters
    ----------
    model : nn.Module
        The model to compress.
    recipe : str or list[Stage]
        Either a recipe name (e.g. "balanced"), a path to a YAML file,
        or a list of Stage instances.
    """
    if isinstance(recipe, (list, tuple)):
        pipeline = Pipeline(list(recipe))
    else:
        pipeline = build_pipeline_from_recipe(recipe)

    return pipeline.run(
        model,
        eval_fn=eval_fn,
        calibration_data=calibration_data,
        device=device,
        constraints=constraints,
        input_shape=input_shape,
    )
