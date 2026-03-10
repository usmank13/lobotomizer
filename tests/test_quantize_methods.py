"""Tests for expanded quantization methods in v0.3."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lobotomizer.stages.quantize import (
    Quantize,
    _QUANT_METHODS,
    available_methods,
    register_quant_method,
    QuantMethod,
)
from lobotomizer.stages.base import PipelineContext


class LinearOnly(nn.Module):
    def __init__(self, in_f: int = 64, hidden: int = 128, out: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class TestMethodRegistry:
    def test_available_methods(self) -> None:
        methods = available_methods()
        assert "dynamic_int8" in methods
        assert "static_int8" in methods
        assert "int4_weight_only" in methods
        assert "gptq" in methods
        assert "awq" in methods

    def test_register_custom_method(self) -> None:
        @register_quant_method("test_custom")
        class CustomMethod(QuantMethod):
            def apply(self, model, context):
                return model

        assert "test_custom" in _QUANT_METHODS
        # Cleanup
        del _QUANT_METHODS["test_custom"]

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown quantization method"):
            Quantize(method="nonexistent_method")


class TestBackwardCompatibility:
    def test_dynamic_alias(self) -> None:
        stage = Quantize(method="dynamic")
        assert stage._method_name == "dynamic_int8"

    def test_static_alias(self) -> None:
        stage = Quantize(method="static")
        assert stage._method_name == "static_int8"

    def test_default_is_dynamic(self) -> None:
        stage = Quantize()
        assert stage._method_name == "dynamic_int8"


class TestDynamicInt8:
    def test_apply(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="dynamic_int8")
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)
        out = result(torch.randn(2, 64))
        assert out.shape == (2, 10)


class TestInt4WeightOnly:
    def test_apply_with_compatible_dims(self) -> None:
        # INT4 needs in_features divisible by 32
        model = LinearOnly(in_f=64, hidden=128, out=10)
        stage = Quantize(method="int4_weight_only")
        ctx = PipelineContext(original_model=model)
        try:
            result = stage.apply(model, ctx)
            out = result(torch.randn(2, 64))
            assert out.shape == (2, 10)
        except ImportError:
            pytest.skip("INT4 weight-only requires additional dependencies (mslk)")

    def test_validate_warns_bad_dims(self) -> None:
        model = nn.Sequential(nn.Linear(13, 10))
        stage = Quantize(method="int4_weight_only")
        warnings = stage.validate(model)
        assert any("not divisible by 32" in w for w in warnings)

    def test_estimate_impact(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="int4_weight_only")
        impact = stage.estimate_impact(model)
        assert impact["compression_ratio"] == 0.125
        assert impact["method"] == "int4_weight_only"


class TestGPTQ:
    def test_missing_dep_raises(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="gptq")
        ctx = PipelineContext(original_model=model, calibration_data=[torch.randn(2, 64)])
        with pytest.raises((ImportError, TypeError)):
            stage.apply(model, ctx)

    def test_validate_warns_missing_dep(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="gptq")
        warnings = stage.validate(model)
        assert any("auto-gptq" in w.lower() for w in warnings)

    def test_validate_warns_non_hf_model(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="gptq")
        warnings = stage.validate(model)
        assert any("huggingface" in w.lower() for w in warnings)

    def test_kwargs_pass_through(self) -> None:
        stage = Quantize(method="gptq", bits=3, group_size=64)
        assert stage._quant_method.kwargs["bits"] == 3
        assert stage._quant_method.kwargs["group_size"] == 64


class TestAWQ:
    def test_missing_dep_raises(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="awq")
        ctx = PipelineContext(original_model=model, calibration_data=["test text"])
        with pytest.raises((ImportError, TypeError)):
            stage.apply(model, ctx)

    def test_validate_warns_missing_dep(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="awq")
        warnings = stage.validate(model)
        assert any("autoawq" in w.lower() for w in warnings)

    def test_estimate_impact(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="awq", w_bit=3)
        impact = stage.estimate_impact(model)
        assert impact["compression_ratio"] == 3 / 32.0


class TestQuantizeName:
    def test_name_includes_method(self) -> None:
        assert Quantize(method="dynamic").name == "quantize(dynamic_int8)"
        assert Quantize(method="int4_weight_only").name == "quantize(int4_weight_only)"
        assert Quantize(method="gptq").name == "quantize(gptq)"
