"""Tests for ONNX export."""
from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

from lobotomizer import compress, LowRank, to_onnx
from lobotomizer.export.onnx import _verify_onnx


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv(x)))
        return self.fc(x.view(x.size(0), -1))


class TestOnnxExport:
    def test_export_mlp(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        out = tmp_path / "model.onnx"
        result = to_onnx(model, out, input_shape=(1, 32), verify=False)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_cnn(self, tmp_path: pathlib.Path) -> None:
        model = SimpleCNN()
        out = tmp_path / "model.onnx"
        result = to_onnx(model, out, input_shape=(1, 3, 8, 8), verify=False)
        assert result.exists()

    def test_export_after_compression(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        compressed = compress(model, recipe=[LowRank(rank_fraction=0.5, min_compression=1.0)])
        out = tmp_path / "compressed.onnx"
        result = to_onnx(compressed.model, out, input_shape=(1, 32), verify=False)
        assert result.exists()

    def test_result_to_onnx(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        result = compress(model, recipe=[LowRank(rank_fraction=0.5, min_compression=1.0)])
        out = tmp_path / "result.onnx"
        path = result.to_onnx(str(out), input_shape=(1, 32), verify=False)
        assert pathlib.Path(path).exists()

    def test_creates_parent_dirs(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        out = tmp_path / "nested" / "dir" / "model.onnx"
        result = to_onnx(model, out, input_shape=(1, 32), verify=False)
        assert result.exists()

    def test_custom_names(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        out = tmp_path / "model.onnx"
        result = to_onnx(
            model, out, input_shape=(1, 32),
            input_names=["features"],
            output_names=["logits"],
            verify=False,
        )
        assert result.exists()

    def test_verify_with_onnx(self, tmp_path: pathlib.Path) -> None:
        """If onnx is installed, verify should pass."""
        model = SimpleMLP()
        out = tmp_path / "model.onnx"
        # verify=True by default; will skip gracefully if onnx not installed
        result = to_onnx(model, out, input_shape=(1, 32), verify=True)
        assert result.exists()

    def test_custom_opset(self, tmp_path: pathlib.Path) -> None:
        model = SimpleMLP()
        out = tmp_path / "model.onnx"
        result = to_onnx(model, out, input_shape=(1, 32), opset_version=13, verify=False)
        assert result.exists()
