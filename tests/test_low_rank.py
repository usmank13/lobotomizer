"""Tests for LowRank decomposition stage."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lobotomizer import LowRank, Pipeline, compress
from lobotomizer.stages.base import PipelineContext


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class WideMLP(nn.Module):
    """MLP with large hidden layer (good SVD compression candidate)."""
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class WideConvNet(nn.Module):
    """Conv net with large channels (good candidate)."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TinyMLP(nn.Module):
    """Very small MLP where decomposition won't help."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Linear decomposition tests
# ---------------------------------------------------------------------------

class TestLowRankLinear:
    def test_decomposes_wide_layer(self) -> None:
        model = WideMLP()
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)  # force decomposition
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        # fc1 should be replaced with Sequential(V, U)
        assert isinstance(result.fc1, nn.Sequential)
        assert len(result.fc1) == 2
        # V: (64 → rank), U: (rank → 256)
        v_layer, u_layer = result.fc1[0], result.fc1[1]
        assert isinstance(v_layer, nn.Linear)
        assert isinstance(u_layer, nn.Linear)
        assert v_layer.in_features == 64
        assert u_layer.out_features == 256
        assert v_layer.out_features == u_layer.in_features  # rank matches

    def test_forward_works_after_decomposition(self) -> None:
        model = WideMLP()
        stage = LowRank(rank_fraction=0.5, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        x = torch.randn(4, 64)
        out = result(x)
        assert out.shape == (4, 10)

    def test_reduces_params(self) -> None:
        model = WideMLP()
        params_before = sum(p.numel() for p in model.parameters())
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)
        params_after = sum(p.numel() for p in result.parameters())
        assert params_after < params_before

    def test_approximation_quality(self) -> None:
        """High rank fraction should produce close approximation."""
        model = WideMLP()
        x = torch.randn(8, 64)
        original_out = model(x).detach()

        stage = LowRank(rank_fraction=0.9, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)
        approx_out = result(x).detach()

        # Should be reasonably close (not exact due to truncation)
        rel_error = (original_out - approx_out).norm() / original_out.norm()
        assert rel_error < 0.5  # generous bound

    def test_skips_small_layers(self) -> None:
        """Layers where decomposition doesn't save params should be skipped."""
        model = TinyMLP()
        params_before = sum(p.numel() for p in model.parameters())
        stage = LowRank(rank_fraction=0.5, min_compression=0.9)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)
        params_after = sum(p.numel() for p in result.parameters())
        # Should be unchanged (4x4 matrix, rank 2 → 4*2 + 2*4 = 16 ≥ 16)
        assert params_after == params_before

    def test_preserves_bias(self) -> None:
        model = WideMLP()
        assert model.fc1.bias is not None
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        # Bias should be on the U (second) layer
        v_layer, u_layer = result.fc1[0], result.fc1[1]
        assert v_layer.bias is None
        assert u_layer.bias is not None

    def test_no_bias_layer(self) -> None:
        model = nn.Sequential(
            nn.Linear(64, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        x = torch.randn(2, 64)
        out = result(x)
        assert out.shape == (2, 10)

    def test_energy_criterion(self) -> None:
        model = WideMLP()
        stage = LowRank(rank_fraction=0.9, criterion="energy", min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        x = torch.randn(4, 64)
        out = result(x)
        assert out.shape == (4, 10)

    def test_exclude_layers(self) -> None:
        model = WideMLP()
        stage = LowRank(rank_fraction=0.25, exclude_layers=["fc1"], min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        # fc1 should NOT be decomposed
        assert isinstance(result.fc1, nn.Linear)


# ---------------------------------------------------------------------------
# Conv2d decomposition tests
# ---------------------------------------------------------------------------

class TestLowRankConv2d:
    def test_decomposes_conv_layer(self) -> None:
        model = WideConvNet()
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        # conv2 (64→128, 3×3) should be decomposed
        assert isinstance(result.conv2, nn.Sequential)
        v_conv, u_conv = result.conv2[0], result.conv2[1]
        assert isinstance(v_conv, nn.Conv2d)
        assert isinstance(u_conv, nn.Conv2d)
        assert v_conv.kernel_size == (3, 3)
        assert u_conv.kernel_size == (1, 1)

    def test_conv_forward_works(self) -> None:
        model = WideConvNet()
        stage = LowRank(rank_fraction=0.5, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        x = torch.randn(2, 3, 8, 8)
        out = result(x)
        assert out.shape == (2, 10)

    def test_conv_reduces_params(self) -> None:
        model = WideConvNet()
        params_before = sum(p.numel() for p in model.parameters())
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)
        params_after = sum(p.numel() for p in result.parameters())
        assert params_after < params_before

    def test_conv_preserves_stride_padding(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
        )
        stage = LowRank(rank_fraction=0.25, min_compression=1.0)
        ctx = PipelineContext(original_model=model)
        result = stage.apply(model, ctx)

        # If decomposed, first conv should have original stride/padding
        if isinstance(result[0], nn.Sequential):
            v_conv = result[0][0]
            assert v_conv.stride == (2, 2)
            assert v_conv.padding == (1, 1)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestLowRankIntegration:
    def test_pipeline_integration(self) -> None:
        model = WideMLP()
        result = compress(model, recipe=[LowRank(rank_fraction=0.25, min_compression=1.0)])
        assert result.model is not None
        x = torch.randn(1, 64)
        out = result.model(x)
        assert out.shape == (1, 10)

    def test_combined_with_quantize(self) -> None:
        from lobotomizer import Quantize
        model = WideMLP()
        result = compress(model, recipe=[
            LowRank(rank_fraction=0.5, min_compression=1.0),
            Quantize(method="dynamic"),
        ])
        assert result.model is not None
        x = torch.randn(1, 64)
        out = result.model(x)
        assert out.shape == (1, 10)

    def test_recipe_yaml_roundtrip(self) -> None:
        """LowRank can be instantiated from recipe config."""
        from lobotomizer.core.recipe import build_pipeline_from_recipe
        recipe = {
            "stages": [
                {"type": "low_rank", "rank_fraction": 0.3, "criterion": "energy"},
            ]
        }
        pipeline = build_pipeline_from_recipe(recipe)
        assert len(pipeline.stages) == 1
        assert "low_rank" in pipeline.stages[0].name

    def test_invalid_rank_fraction(self) -> None:
        with pytest.raises(ValueError, match="rank_fraction"):
            LowRank(rank_fraction=0.0)
        with pytest.raises(ValueError, match="rank_fraction"):
            LowRank(rank_fraction=1.5)

    def test_invalid_criterion(self) -> None:
        with pytest.raises(ValueError, match="criterion"):
            LowRank(criterion="random")

    def test_validate_no_layers(self) -> None:
        model = nn.BatchNorm2d(16)
        stage = LowRank()
        warnings = stage.validate(model)
        assert any("no linear" in w.lower() for w in warnings)

    def test_name_property(self) -> None:
        stage = LowRank(rank_fraction=0.3, criterion="energy")
        assert "energy" in stage.name
        assert "0.3" in stage.name
