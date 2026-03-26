"""Tests for ONNX export utilities."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from lobotomizer.export.onnx import to_onnx


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x))
        return self.fc(x.view(x.size(0), -1))


class TestToOnnx:
    def test_export_with_input_shape(self, tmp_path) -> None:
        model = SimpleMLP()
        out = to_onnx(model, tmp_path / "model.onnx", input_shape=(1, 16))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_with_dummy_input(self, tmp_path) -> None:
        model = SimpleMLP()
        dummy = torch.randn(2, 16)
        out = to_onnx(model, tmp_path / "model.onnx", dummy_input=dummy)
        assert out.exists()

    def test_export_cnn(self, tmp_path) -> None:
        model = SimpleCNN()
        out = to_onnx(model, tmp_path / "model.onnx", input_shape=(1, 3, 8, 8))
        assert out.exists()

    def test_export_creates_parent_dirs(self, tmp_path) -> None:
        model = SimpleMLP()
        out = to_onnx(model, tmp_path / "sub" / "dir" / "model.onnx", input_shape=(1, 16))
        assert out.exists()

    def test_requires_shape_or_input(self, tmp_path) -> None:
        model = SimpleMLP()
        with pytest.raises(ValueError, match="input_shape or dummy_input"):
            to_onnx(model, tmp_path / "model.onnx")

    def test_custom_names(self, tmp_path) -> None:
        model = SimpleMLP()
        out = to_onnx(
            model, tmp_path / "model.onnx",
            input_shape=(1, 16),
            input_names=["x"],
            output_names=["logits"],
        )
        assert out.exists()

    def test_after_pruning(self, tmp_path) -> None:
        """Export a pruned model."""
        from lobotomizer import compress, Prune
        model = SimpleMLP()
        result = compress(model, recipe=[Prune(method="l1_unstructured", sparsity=0.3)])
        out = to_onnx(result.model, tmp_path / "compressed.onnx", input_shape=(1, 16))
        assert out.exists()

    def test_after_low_rank(self, tmp_path) -> None:
        """Export a low-rank decomposed model."""
        from lobotomizer import compress, LowRank
        model = SimpleMLP()
        result = compress(model, recipe=[LowRank(rank_fraction=0.5, min_compression=1.0)])
        out = to_onnx(result.model, tmp_path / "compressed.onnx", input_shape=(1, 16))
        assert out.exists()

    def test_validate_onnx(self, tmp_path) -> None:
        """Validate exported ONNX if onnx package available."""
        model = SimpleMLP()
        path = to_onnx(model, tmp_path / "model.onnx", input_shape=(1, 16))
        try:
            from lobotomizer.export.onnx import validate_onnx
            assert validate_onnx(path)
        except ImportError:
            pytest.skip("onnx package not installed")
