"""Tests for the CLI."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from lobotomizer.cli import main


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def saved_model(tmp_path: pathlib.Path) -> pathlib.Path:
    """Save a small model to a temp file."""
    model = SmallModel()
    p = tmp_path / "model.pt"
    torch.save(model, p)
    return p


class TestListRecipes:
    def test_list_recipes(self, capsys):
        main(["--list-recipes"])
        out = capsys.readouterr().out
        assert "balanced" in out
        assert "stages:" in out

    def test_list_recipes_no_model_needed(self, capsys):
        """--list-recipes should work without a model argument."""
        main(["--list-recipes"])
        # Should not raise


class TestProfileOnly:
    def test_profile_only(self, saved_model, capsys):
        main([str(saved_model), "--profile-only"])
        out = capsys.readouterr().out
        assert "Parameters:" in out
        assert "Size (MB):" in out

    def test_profile_with_input_shape(self, saved_model, capsys):
        main([str(saved_model), "--profile-only", "--input-shape", "1,32"])
        out = capsys.readouterr().out
        assert "Parameters:" in out


class TestCompression:
    def test_compress_with_recipe(self, saved_model, tmp_path, capsys):
        out_dir = str(tmp_path / "compressed")
        main([str(saved_model), "--recipe", "balanced", "--output", out_dir])
        out = capsys.readouterr().out
        assert "param_count" in out
        assert (pathlib.Path(out_dir) / "model.pt").exists()
        assert (pathlib.Path(out_dir) / "metadata.json").exists()
        meta = json.loads((pathlib.Path(out_dir) / "metadata.json").read_text())
        assert "stages" in meta

    def test_compress_with_explicit_args(self, saved_model, tmp_path, capsys):
        out_dir = str(tmp_path / "out")
        main([str(saved_model), "--prune", "l1_unstructured", "--sparsity", "0.3",
              "--quantize", "dynamic", "--output", out_dir])
        out = capsys.readouterr().out
        assert "param_count" in out
        assert (pathlib.Path(out_dir) / "model.pt").exists()

    def test_compress_prune_only(self, saved_model, capsys):
        main([str(saved_model), "--prune", "random_unstructured", "--sparsity", "0.2"])
        out = capsys.readouterr().out
        assert "param_count" in out


class TestErrors:
    def test_no_model_no_list(self, capsys):
        with pytest.raises(SystemExit):
            main([])

    def test_missing_file(self, capsys):
        with pytest.raises(SystemExit):
            main(["nonexistent.pt", "--recipe", "balanced"])

    def test_no_compression_args(self, saved_model):
        with pytest.raises(SystemExit):
            main([str(saved_model)])

    def test_invalid_prune_method(self, saved_model):
        with pytest.raises(ValueError, match="Unknown pruning method"):
            main([str(saved_model), "--prune", "bad_method"])
