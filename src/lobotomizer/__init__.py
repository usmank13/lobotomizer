"""Lobotomizer — composable model compression for PyTorch."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from lobotomizer.core.pipeline import Pipeline
from lobotomizer.core.recipe import build_pipeline_from_recipe
from lobotomizer.core.result import Result
from lobotomizer.stages.base import Stage
from lobotomizer.stages.prune import Prune
from lobotomizer.stages.quantize import Quantize, available_methods as available_quant_methods
from lobotomizer.stages.distill import Distill
from lobotomizer.stages.low_rank import LowRank
from lobotomizer.stages.structured_prune import StructuredPrune
from lobotomizer.core.registry import register_adapter, register_stage
from lobotomizer.export import to_onnx
from lobotomizer.export import to_onnx

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.utils.data import DataLoader

__version__ = "0.3.0"
__all__ = [
    "Distill",
    "LowRank",
    "Pipeline",
    "Prune",
    "Quantize",
    "Result",
    "Stage",
    "StructuredPrune",
    "compress",
    "register_adapter",
    "register_stage",
    "to_onnx",
    "to_onnx",
]


def compress(
    model: nn.Module,
    recipe: str | list[Stage] = "balanced",
    *,
    eval_fn: Callable | None = None,
    calibration_data: DataLoader | None = None,
    training_data: DataLoader | None = None,
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
    training_data : DataLoader, optional
        Training data for knowledge distillation stages.
    """
    if isinstance(recipe, (list, tuple)):
        pipeline = Pipeline(list(recipe))
    else:
        pipeline = build_pipeline_from_recipe(recipe)

    return pipeline.run(
        model,
        eval_fn=eval_fn,
        calibration_data=calibration_data,
        training_data=training_data,
        device=device,
        constraints=constraints,
        input_shape=input_shape,
    )
