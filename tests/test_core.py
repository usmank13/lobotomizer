"""Tests for core abstractions."""
from __future__ import annotations

import pytest
import torch.nn as nn

from lobotomizer.core.pipeline import ConstraintViolation, Pipeline
from lobotomizer.core.profile import profile_model
from lobotomizer.core.result import Result
from lobotomizer.stages.base import PipelineContext, StageResult


# --- PipelineContext ---

def test_pipeline_context_creation(simple_model):
    ctx = PipelineContext(original_model=simple_model)
    assert ctx.device == "cpu"
    assert ctx.eval_fn is None
    assert ctx.history == []


# --- StageResult.delta ---

def test_stage_result_delta(simple_model):
    sr = StageResult(
        stage_name="test",
        model=simple_model,
        metrics_before={"param_count": 100, "size_mb": 1.0},
        metrics_after={"param_count": 50, "size_mb": 0.5},
    )
    d = sr.delta
    assert d["param_count"] == -50
    assert d["size_mb"] == pytest.approx(-0.5)


def test_stage_result_delta_missing_keys(simple_model):
    sr = StageResult(
        stage_name="test",
        model=simple_model,
        metrics_before={"a": 10},
        metrics_after={"a": 5, "b": 3},
    )
    d = sr.delta
    assert d["a"] == -5
    assert "b" in d


# --- profile_model ---

def test_profile_simple_model(simple_model):
    p = profile_model(simple_model)
    # nn.Linear(32, 16) => 32*16 + 16 = 528 params
    assert p["param_count"] == 528
    assert p["param_count_trainable"] == 528
    assert p["size_mb"] > 0
    assert p["flops"] is None  # no input_shape


def test_profile_with_input_shape(simple_model):
    # FLOPs may be None if no backend available — just ensure no crash
    p = profile_model(simple_model, input_shape=(1, 32))
    assert "flops" in p


# --- Pipeline with identity stage ---

def test_pipeline_identity(simple_model, identity_stage):
    pipe = Pipeline([identity_stage])
    result = pipe.run(simple_model)
    assert isinstance(result, Result)
    assert len(result.stage_results) == 1
    assert result.stage_results[0].stage_name == "identity"


def test_pipeline_does_not_mutate_original(simple_model, identity_stage):
    import copy, torch
    original_sd = copy.deepcopy(simple_model.state_dict())
    pipe = Pipeline([identity_stage])
    pipe.run(simple_model)
    for k in original_sd:
        assert torch.equal(original_sd[k], simple_model.state_dict()[k])


# --- Result summary ---

def test_result_summary(simple_model, identity_stage):
    pipe = Pipeline([identity_stage])
    result = pipe.run(simple_model)
    s = result.summary()
    assert "param_count" in s
    assert "┌" in s  # box-drawing chars


# --- Constraint violation ---

def test_pipeline_constraint_violation(simple_model, identity_stage):
    def bad_eval(model):
        return 0.1  # below threshold

    pipe = Pipeline([identity_stage])
    with pytest.raises(ConstraintViolation, match="violated constraint"):
        pipe.run(
            simple_model,
            eval_fn=bad_eval,
            constraints={"min_eval_score": 0.5},
        )
