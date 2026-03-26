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


class SimpleCNN(nn.Module):
    """Two conv layers with BatchNorm, then a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ConvOnlyCNN(nn.Module):
    """Conv layers without BatchNorm."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
        """Legacy test: protect_output=False to verify all layers get pruned."""
        model = TwoLayerMLP()
        assert model.fc1.out_features == 128
        assert model.fc2.in_features == 128

        stage = StructuredPrune(sparsity=0.5, criterion="l1", protect_output=False)
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
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(4, 64)
        out = pruned(x)
        assert out.shape == (4, 5)  # output layer also pruned

    def test_l2_criterion(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.25, criterion="l2", protect_output=False)
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
        stage = StructuredPrune(sparsity=0.5, criterion="l1", protect_output=False)
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
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.fc.out_features == 4
        assert pruned.fc.in_features == 16  # input unchanged

        x = torch.randn(3, 16)
        out = pruned(x)
        assert out.shape == (3, 4)

    def test_bias_pruned_correctly(self) -> None:
        model = TwoLayerMLP()
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.fc1.bias is not None
        assert pruned.fc1.bias.shape == (64,)
        assert pruned.fc2.bias is not None
        assert pruned.fc2.bias.shape == (5,)  # output layer also pruned

    def test_fewer_params_after_pruning(self) -> None:
        model = TwoLayerMLP()
        params_before = sum(p.numel() for p in model.parameters())
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
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
        result = compress(model, recipe=[StructuredPrune(sparsity=0.3, protect_output=False)])
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
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 5)

    def test_protect_output_preserves_final_layer(self) -> None:
        """Output layer dimensions should be preserved by default."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        stage = StructuredPrune(sparsity=0.5, protect_output=True)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 32)
        out = pruned(x)
        # Output dim must remain 10 (the model's interface)
        assert out.shape == (2, 10)
        # But the hidden layer should still be pruned
        assert pruned[0].out_features == 32  # 64 * 0.5

    def test_protect_output_is_default(self) -> None:
        """protect_output=True is the default behavior."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 32)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_protect_output_disabled(self) -> None:
        """protect_output=False allows pruning the output layer."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # Output layer should be pruned too
        assert pruned[2].out_features == 5  # 10 * 0.5

    def test_exclude_layers_by_name(self) -> None:
        """User can exclude specific layers from pruning."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        # Exclude the middle layer
        stage = StructuredPrune(
            sparsity=0.5,
            protect_output=False,
            exclude_layers=["2"],
        )
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # Layer "2" (second Linear) should keep its original out_features
        assert pruned[2].out_features == 32
        # First layer should be pruned
        assert pruned[0].out_features == 32  # 64 * 0.5

    def test_exclude_layers_nonexistent_warns(self) -> None:
        """Excluding a nonexistent layer logs a warning but doesn't crash."""
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
        stage = StructuredPrune(sparsity=0.5, exclude_layers=["nonexistent"])
        ctx = PipelineContext(original_model=model)
        # Should not raise
        pruned = stage.apply(model, ctx)
        assert pruned is model

    def test_protect_output_three_layer(self) -> None:
        """In a 3-layer chain, only the final output is protected."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        stage = StructuredPrune(sparsity=0.5)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 64)
        out = pruned(x)
        # Output preserved
        assert out.shape == (2, 10)
        # Hidden layers pruned
        assert pruned[0].out_features == 64  # 128 * 0.5
        assert pruned[2].out_features == 32  # 64 * 0.5


class TestConv2dStructuredPrune:
    def test_conv_reduces_channels(self) -> None:
        model = ConvOnlyCNN()
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # First conv: out_channels 16 → 8
        assert pruned.net[0].out_channels == 8
        # Second conv: in_channels matches, out_channels also pruned
        assert pruned.net[2].in_channels == 8
        assert pruned.net[2].out_channels == 16

    def test_conv_forward_works(self) -> None:
        model = ConvOnlyCNN()
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 3, 8, 8)
        out = pruned(x)
        assert out.shape == (2, 16, 8, 8)

    def test_conv_with_batchnorm(self) -> None:
        model = SimpleCNN()
        stage = StructuredPrune(sparsity=0.5, protect_output=True)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # First conv: 32 → 16 channels
        conv1 = pruned.features[0]
        bn1 = pruned.features[1]
        conv2 = pruned.features[3]
        bn2 = pruned.features[4]

        assert conv1.out_channels == 16
        assert bn1.num_features == 16
        assert bn1.weight.shape[0] == 16
        assert bn1.running_mean.shape[0] == 16
        assert conv2.in_channels == 16

        # Second conv is output of features Sequential → protected
        assert conv2.out_channels == 64
        assert bn2.num_features == 64
        # Classifier input stays matching conv2 output (both unchanged)
        assert pruned.classifier.in_features == 64

    def test_conv_batchnorm_forward(self) -> None:
        model = SimpleCNN()
        stage = StructuredPrune(sparsity=0.5, protect_output=True)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        x = torch.randn(2, 3, 8, 8)
        out = pruned(x)
        assert out.shape == (2, 10)

    def test_conv_fewer_params(self) -> None:
        model = SimpleCNN()
        params_before = sum(p.numel() for p in model.parameters())
        stage = StructuredPrune(sparsity=0.5, protect_output=True)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)
        params_after = sum(p.numel() for p in pruned.parameters())
        assert params_after < params_before

    def test_conv_protect_output(self) -> None:
        """Last conv (no downstream conv) should be protected."""
        model = ConvOnlyCNN()
        stage = StructuredPrune(sparsity=0.5, protect_output=True)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        # First conv pruned
        assert pruned.net[0].out_channels == 8
        # Second conv (output) protected
        assert pruned.net[2].out_channels == 32
        assert pruned.net[2].in_channels == 8

    def test_conv_l2_criterion(self) -> None:
        model = ConvOnlyCNN()
        stage = StructuredPrune(sparsity=0.25, criterion="l2", protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned.net[0].out_channels == 12  # 16 * 0.75
        assert pruned.net[2].in_channels == 12

    def test_validate_reports_conv_layers(self) -> None:
        """Conv-only model should pass validation."""
        model = ConvOnlyCNN()
        stage = StructuredPrune()
        warnings = stage.validate(model)
        assert len(warnings) == 0

    def test_conv_no_bias(self) -> None:
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
        )
        stage = StructuredPrune(sparsity=0.5, protect_output=False)
        ctx = PipelineContext(original_model=model)
        pruned = stage.apply(model, ctx)

        assert pruned[0].out_channels == 8
        assert pruned[0].bias is None
        x = torch.randn(1, 3, 4, 4)
        out = pruned(x)
        assert out.shape == (1, 16, 4, 4)

    def test_pipeline_integration_conv(self) -> None:
        model = SimpleCNN()
        result = compress(model, recipe=[StructuredPrune(sparsity=0.3)])
        assert result.model is not None
        x = torch.randn(1, 3, 8, 8)
        out = result.model(x)
        assert out.shape == (1, 10)
