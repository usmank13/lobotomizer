"""Pipeline composition engine."""
from __future__ import annotations

import copy
import warnings
from typing import Callable

import torch.nn as nn
from torch.utils.data import DataLoader

from lobotomizer.core.profile import profile_model
from lobotomizer.core.result import Result
from lobotomizer.stages.base import PipelineContext, Stage, StageResult


class ConstraintViolation(Exception):
    """Raised when a pipeline constraint is violated."""


class Pipeline:
    """Composable compression pipeline."""

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = list(stages)

    def run(
        self,
        model: nn.Module,
        *,
        eval_fn: Callable | None = None,
        calibration_data: DataLoader | None = None,
        device: str = "cpu",
        constraints: dict | None = None,
        input_shape: tuple | None = None,
    ) -> Result:
        """Run all stages sequentially and return a :class:`Result`."""
        constraints = constraints or {}

        # 1. Profile original
        profile_before = profile_model(model, input_shape=input_shape, device=device)

        # 2. Validate all stages upfront
        for stage in self.stages:
            warns = stage.validate(model)
            for w in warns:
                warnings.warn(f"[{stage.name}] {w}", stacklevel=2)

        # 3. Deep-copy so we never mutate the original
        working_model = copy.deepcopy(model).to(device)

        context = PipelineContext(
            original_model=model,
            eval_fn=eval_fn,
            target_constraints=constraints,
            calibration_data=calibration_data,
            device=device,
        )

        # 4. Run each stage
        for stage in self.stages:
            before = profile_model(working_model, input_shape=input_shape, device=device)
            working_model = stage.apply(working_model, context)
            after = profile_model(working_model, input_shape=input_shape, device=device)

            sr = StageResult(
                stage_name=stage.name,
                model=working_model,
                metrics_before=before,
                metrics_after=after,
            )
            context.history.append(sr)

            # Check constraints
            _check_constraints(constraints, eval_fn, working_model, stage.name)

        # 5. Final profile & result
        profile_after = profile_model(working_model, input_shape=input_shape, device=device)
        return Result(
            original_model=model,
            compressed_model=working_model,
            stage_results=context.history,
            profile_before=profile_before,
            profile_after=profile_after,
        )


def _check_constraints(
    constraints: dict,
    eval_fn: Callable | None,
    model: nn.Module,
    stage_name: str,
) -> None:
    """Abort if quality drops below the specified threshold."""
    min_quality = constraints.get("min_eval_score")
    if min_quality is not None and eval_fn is not None:
        score = eval_fn(model)
        if score < min_quality:
            raise ConstraintViolation(
                f"Stage '{stage_name}' violated constraint: "
                f"eval score {score} < minimum {min_quality}"
            )
