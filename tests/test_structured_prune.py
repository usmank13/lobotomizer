"""Tests for StructuredPrune stage."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lobotomizer import StructuredPrune, Pipeline, compress
from lobotomizer.stages.base import PipelineContext


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class TwoLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ThreeLayerMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStructuredPrune:
    def test_reduces_dimensions(self) -> None:
        model = TwoLayerMLP()
        assert model.fc1.out_features == 128
        assert model.fc2.in_features == 128

        stage = StructuredPrune(sparsity=0.5, criterion="l1")
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # fc1 output and fc2 input should both shrink to ~64
        assert pruned.fc1.out_features == 64
        assert pruned.fc2.in_features == 64
        assert pruned.fc1.weight.shape == (64, 64)
        # fc2 output also pruned (all layers get output pruning)
        assert pruned.fc2.out_features == 5
        assert pruned.fc2.weight.shape == (5, 64)

    def test_forward_still_works(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(4, 64)
        out = pruned(x)
        assert out.shape == (4, 5)  # output layer also pruned

    def test_l2_criterion(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.25, criterion="l2")
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.fc1.out_features == 96
        assert pruned.fc2.in_features == 96

        x = torch.randn(2, 64)
        out = pruned(x)
        # fc2 output: 10 * 0.75 = 7 (rounded)
        assert out.shape[0] == 2
        assert out.shape[1] < 10  # output also pruned

    def test_three_layer_propagation(self) -> None:
        model = ThreeLayerMLP()
        stage = StructuredPrune(sparsity=0.5, criterion="l1")
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # All internal dimensions should shrink
        layers = [m for m in pruned.modules() if isinstance(m, nn.Linear)]
        # Layer 0: (32->32), layer 1: (32->32), layer 2: (32->5) — approximately
        # Actually: layer 0 out=32, layer 1 in=32 out=32, layer 2 in=32 out=5
        assert layers[0].out_features == 32  # 64 * 0.5
        assert layers[1].in_features == 32
        assert layers[1].out_features == 32
        assert layers[2].in_features == 32

        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 5)  # last layer also pruned from 10

    def test_single_layer_no_propagation_needed(self) -> None:
        model = SingleLinear()
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.fc.out_features == 4
        assert pruned.fc.in_features == 16  # input unchanged

        x = torch.randn(3, 16)
        out = pruned(x)
        assert out.shape == (3, 4)

    def test_bias_pruned_correctly(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.fc1.bias is not None
        assert pruned.fc1.bias.shape == (64,)
        assert pruned.fc2.bias is not None
        assert pruned.fc2.bias.shape == (5,)  # output layer also pruned

    def test_fewer_params_after_pruning(self) -> None:
        model = TwoLayerMLP()
        params_before = sum(p.numel() for p in model.parameters())
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)
        params_after = sum(p.numel() for p in pruned.parameters())

        assert params_after < params_before

    def test_invalid_sparsity(self) -> None:
        with pytest.raises(ValueError, match="Sparsity"):
            StructuredPrune(sparsity=1.0)
        with pytest.raises(ValueError, match="Sparsity"):
            StructuredPrune(sparsity=-0.1)

    def test_invalid_criterion(self) -> None:
        with pytest.raises(ValueError, match="Criterion"):
            StructuredPrune(criterion="linf")

    def test_validate_no_linear_layers(self) -> None:
        model = nn.BatchNorm2d(16)
        stage = StructuredPrune()
        warnings = stage.validate(model)
        assert any("no linear" in w.lower() for w in warnings)

    def test_estimate_impact(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.5)
        impact = stage.estimate_impact(model)
        assert impact["estimated_removed"] > 0

    def test_name_property(self) -> None:
        stage = StructuredPrune(sparsity=0.4, criterion="l2")
        assert "l2" in stage.name
        assert "0.4" in stage.name

    def test_pipeline_integration(self) -> None:
        model = TwoLayerMLP()
        result = compress(model, recipe=[StructuredPrune(sparsity=0.3)])
        assert result.model is not None
        x = torch.randn(1, 64)
        out = result.model(x)
        assert out.shape[1] < 10  # all layers get output pruning

    def test_no_bias_layer(self) -> None:
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
        )
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 5)
