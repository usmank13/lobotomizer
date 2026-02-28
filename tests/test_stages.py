"""Tests for Prune, Quantize stages, recipe loading, and compress() API."""
from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn

from lobotomizer import Pipeline, Prune, Quantize, compress
from lobotomizer.core.recipe import build_pipeline_from_recipe
from lobotomizer.stages.base import PipelineContext


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class SmallConvNet(nn.Module):
    """Tiny CNN for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class LinearOnly(nn.Module):
    """Model with only Linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _model_size_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def _count_zeros(model: nn.Module) -> tuple[int, int]:
    """Return (total_elements, zero_elements) across all params."""
    total = 0
    zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return total, zeros


# ---------------------------------------------------------------------------
# Prune tests
# ---------------------------------------------------------------------------

class TestPrune:
    def test_l1_unstructured_sparsity(self) -> None:
        model = SmallConvNet()
        _, zeros_before = _count_zeros(model)
        stage = Prune(method="l1_unstructured", sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)
        total, zeros_after = _count_zeros(pruned)
        # Should have roughly 50% zeros in prunable layers
        assert zeros_after > zeros_before
        sparsity = zeros_after / total
        assert sparsity > 0.3  # should be ~0.5 but some layers have few params

    def test_random_unstructured(self) -> None:
        model = LinearOnly()
        stage = Prune(method="random_unstructured", sparsity=0.4)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)
        total, zeros = _count_zeros(pruned)
        assert zeros > 0

    def test_l1_structured(self) -> None:
        model = SmallConvNet()
        stage = Prune(method="l1_structured", sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)
        # Structured pruning zeros out entire channels
        total, zeros = _count_zeros(pruned)
        assert zeros > 0

    def test_validate_no_prunable_layers(self) -> None:
        model = nn.BatchNorm2d(16)
        stage = Prune()
        warnings = stage.validate(model)
        assert any("no prunable" in w.lower() for w in warnings)

    def test_estimate_impact(self) -> None:
        model = SmallConvNet()
        stage = Prune(sparsity=0.5)
        impact = stage.estimate_impact(model)
        assert impact["estimated_removed"] > 0
        assert impact["estimated_remaining"] < impact["total_params"]

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown pruning method"):
            Prune(method="nonexistent")

    def test_invalid_sparsity(self) -> None:
        with pytest.raises(ValueError, match="Sparsity"):
            Prune(sparsity=1.5)


# ---------------------------------------------------------------------------
# Quantize tests
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_dynamic_reduces_size(self) -> None:
        model = LinearOnly()
        size_before = _model_size_bytes(model)
        stage = Quantize(method="dynamic")
        ctx = PipelineContext(original_model=model)
        quantized = stage.apply(model, ctx)
        size_after = _model_size_bytes(quantized)
        assert size_after < size_before

    def test_dynamic_output_shape(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="dynamic")
        ctx = PipelineContext(original_model=model)
        quantized = stage.apply(model, ctx)
        x = torch.randn(2, 64)
        out = quantized(x)
        assert out.shape == (2, 10)

    def test_validate_warns_static(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="static")
        warnings = stage.validate(model)
        assert any("calibration" in w.lower() for w in warnings)

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown quantization"):
            Quantize(method="bad")

    def test_estimate_impact(self) -> None:
        model = LinearOnly()
        stage = Quantize(method="dynamic")
        impact = stage.estimate_impact(model)
        assert impact["estimated_bytes_saved"] > 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_prune_then_quantize(self) -> None:
        model = LinearOnly()
        result = Pipeline([
            Prune(method="l1_unstructured", sparsity=0.3),
            Quantize(method="dynamic"),
        ]).run(model)
        assert result.model is not None
        assert len(result.stage_results) == 2
        # Size should decrease
        assert result.profile_after["size_mb"] <= result.profile_before["size_mb"]

    def test_summary(self) -> None:
        model = LinearOnly()
        result = compress(model, recipe=[Prune(sparsity=0.2), Quantize()])
        summary = result.summary()
        assert "param_count" in summary


# ---------------------------------------------------------------------------
# Recipe loading
# ---------------------------------------------------------------------------

class TestRecipe:
    def test_load_balanced(self) -> None:
        pipeline = build_pipeline_from_recipe("balanced")
        assert len(pipeline.stages) == 2

    def test_balanced_recipe_runs(self) -> None:
        model = LinearOnly()
        result = compress(model, recipe="balanced")
        assert result.model is not None
        x = torch.randn(1, 64)
        out = result.model(x)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# compress() API
# ---------------------------------------------------------------------------

class TestCompress:
    def test_compress_with_recipe_name(self) -> None:
        model = SmallConvNet()
        result = compress(model, recipe="balanced")
        # Allow small overhead from quantization wrappers on tiny models
        assert result.profile_after["size_mb"] <= result.profile_before["size_mb"] * 1.1

    def test_compress_with_stage_list(self) -> None:
        model = LinearOnly()
        result = compress(model, recipe=[Quantize(method="dynamic")])
        assert result.model is not None

    def test_compress_convnet(self) -> None:
        model = SmallConvNet()
        result = compress(model, recipe=[
            Prune(method="l1_unstructured", sparsity=0.5),
            Quantize(method="dynamic"),
        ])
        x = torch.randn(1, 3, 32, 32)
        out = result.model(x)
        assert out.shape == (1, 10)


class TestSparsityReport:
    """Tests for Result.sparsity_report()."""

    def test_report_after_pruning(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        result = compress(model, recipe=[Prune(method="l1_unstructured", sparsity=0.5)])
        report = result.sparsity_report()
        assert "50.0%" in report
        assert "TOTAL" in report

    def test_report_after_quantize(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        result = compress(model, recipe=[Quantize(method="dynamic")])
        report = result.sparsity_report()
        assert "TOTAL" in report
        # Quantize without pruning should show ~0% sparsity
        # Quantize without pruning should show very low sparsity
        assert "TOTAL" in report  # already checked above
        # Sparsity should be under 5% (just natural zeros from quantization)
        import re
        total_line = [l for l in report.split("\n") if "TOTAL" in l][0]
        match = re.search(r"(\d+\.\d+)%", total_line)
        assert match and float(match.group(1)) < 5.0

    def test_report_no_double_counting(self):
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        result = compress(model, recipe="balanced")
        report = result.sparsity_report()
        # Should not have duplicate layers
        lines = [l for l in report.split("\n") if "LinearPackedParams" in l or "Linear " in l]
        # At most 2 weight-bearing layers (not 4)
        assert len(lines) <= 2
