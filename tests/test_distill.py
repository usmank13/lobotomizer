"""Tests for knowledge distillation stage."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lobotomizer.core.pipeline import Pipeline
from lobotomizer.core.recipe import build_pipeline_from_recipe
from lobotomizer.stages.adapters import (
    FeatureAdapter,
    FunctionalAdapter,
    resolve_adapter,
)
from lobotomizer.stages.base import PipelineContext
from lobotomizer.stages.distill import Distill, _auto_match_layers
from lobotomizer.stages.distill_losses import (
    CombinedKDLoss,
    FeatureMatchingLoss,
    KDLoss,
)
from lobotomizer.stages.hooks import ForwardHookManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SmallMLP(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def mlp():
    torch.manual_seed(42)
    return SmallMLP()


@pytest.fixture
def train_loader():
    torch.manual_seed(0)
    x = torch.randn(80, 32)
    y = torch.randint(0, 10, (80,))
    return DataLoader(TensorDataset(x, y), batch_size=16)


# ---------------------------------------------------------------------------
# KDLoss
# ---------------------------------------------------------------------------


class TestKDLoss:
    def test_gradient_flow(self):
        loss_fn = KDLoss(temperature=4.0)
        s = torch.randn(4, 10, requires_grad=True)
        t = torch.randn(4, 10)
        loss = loss_fn(s, t)
        loss.backward()
        assert s.grad is not None
        assert loss.item() >= 0

    def test_identical_inputs_low_loss(self):
        loss_fn = KDLoss(temperature=4.0)
        logits = torch.randn(4, 10)
        loss = loss_fn(logits, logits)
        assert loss.item() < 1e-5


# ---------------------------------------------------------------------------
# FeatureMatchingLoss
# ---------------------------------------------------------------------------


class TestFeatureMatchingLoss:
    def test_matched_dims(self):
        adapters = {"layer": nn.Identity()}
        loss_fn = FeatureMatchingLoss(adapters)
        s = {"layer": torch.randn(4, 16, requires_grad=True)}
        t = {"layer": torch.randn(4, 16)}
        loss = loss_fn(s, t)
        loss.backward()
        assert s["layer"].grad is not None

    def test_mismatched_dims(self):
        adapter = FeatureAdapter(8, 16)
        loss_fn = FeatureMatchingLoss({"layer": adapter})
        s = {"layer": torch.randn(4, 8, requires_grad=True)}
        t = {"layer": torch.randn(4, 16)}
        loss = loss_fn(s, t)
        loss.backward()
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# ForwardHookManager
# ---------------------------------------------------------------------------


class TestForwardHookManager:
    def test_lifecycle(self, mlp):
        mgr = ForwardHookManager()
        mgr.register(mlp, ["fc1", "fc2"])

        x = torch.randn(2, 32)
        mlp(x)

        feats = mgr.pop_features()
        assert "fc1" in feats
        assert "fc2" in feats
        assert feats["fc1"].shape == (2, 64)

        # After pop, buffer is empty
        assert mgr.pop_features() == {}

        mgr.remove_hooks()
        mlp(x)
        assert mgr.pop_features() == {}

    def test_invalid_layer(self, mlp):
        mgr = ForwardHookManager()
        with pytest.raises(KeyError, match="nonexistent"):
            mgr.register(mlp, ["nonexistent"])


# ---------------------------------------------------------------------------
# resolve_adapter
# ---------------------------------------------------------------------------


class TestResolveAdapter:
    def test_none_auto(self):
        a = resolve_adapter(None, 8, 16)
        assert isinstance(a, FeatureAdapter)
        out = a(torch.randn(2, 8))
        assert out.shape == (2, 16)

    def test_none_identity(self):
        a = resolve_adapter(None, 16, 16)
        assert isinstance(a, FeatureAdapter)
        x = torch.randn(2, 16)
        assert torch.equal(a(x), x)

    def test_type(self):
        a = resolve_adapter(FeatureAdapter, 8, 16)
        assert isinstance(a, FeatureAdapter)

    def test_module_direct(self):
        m = nn.Linear(8, 16)
        a = resolve_adapter(m, 8, 16)
        assert a is m

    def test_callable(self):
        fn = lambda x: x * 2
        a = resolve_adapter(fn, 8, 8)
        assert isinstance(a, FunctionalAdapter)

    def test_str_unknown(self):
        with pytest.raises(KeyError):
            resolve_adapter("nonexistent", 8, 8)


# ---------------------------------------------------------------------------
# Distill stage integration
# ---------------------------------------------------------------------------


class TestDistillIntegration:
    def test_logit_distill(self, mlp, train_loader):
        teacher = SmallMLP()
        teacher.load_state_dict(mlp.state_dict())

        stage = Distill(teacher=teacher, method="logit", epochs=2, lr=1e-3, log_every=0)
        ctx = PipelineContext(
            original_model=teacher,
            training_data=train_loader,
        )
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_feature_distill(self, mlp, train_loader):
        teacher = SmallMLP()
        teacher.load_state_dict(mlp.state_dict())

        stage = Distill(
            teacher=teacher,
            method="feature",
            feature_layers={"fc1": "fc1", "fc2": "fc2"},
            epochs=2,
            lr=1e-3,
            log_every=0,
        )
        ctx = PipelineContext(original_model=teacher, training_data=train_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_both_distill(self, mlp, train_loader):
        teacher = SmallMLP()
        teacher.load_state_dict(mlp.state_dict())

        stage = Distill(
            teacher=teacher,
            method="both",
            feature_layers={"fc1": "fc1"},
            epochs=1,
            lr=1e-3,
            log_every=0,
        )
        ctx = PipelineContext(original_model=teacher, training_data=train_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_loss_decreases(self, train_loader):
        """Verify that distillation training actually reduces loss."""
        torch.manual_seed(42)
        teacher = SmallMLP()
        torch.manual_seed(99)
        student = SmallMLP()

        # Compute initial KD loss
        kd = KDLoss(temperature=4.0)
        initial_losses = []
        student.eval()
        teacher.eval()
        with torch.no_grad():
            for batch in train_loader:
                x, _ = batch
                initial_losses.append(kd(student(x), teacher(x)).item())
        avg_initial = sum(initial_losses) / len(initial_losses)

        # Distill
        stage = Distill(teacher=teacher, method="logit", epochs=5, lr=1e-3, log_every=0)
        ctx = PipelineContext(original_model=teacher, training_data=train_loader)
        student = stage.apply(student, ctx)

        # Compute final KD loss
        final_losses = []
        student.eval()
        with torch.no_grad():
            for batch in train_loader:
                x, _ = batch
                final_losses.append(kd(student(x), teacher(x)).item())
        avg_final = sum(final_losses) / len(final_losses)

        assert avg_final < avg_initial

    def test_pipeline_prune_then_distill(self, train_loader):
        from lobotomizer.stages.structured_prune import StructuredPrune

        torch.manual_seed(42)
        # Use feature distillation since pruning changes output dims
        model = SmallMLP()

        pipeline = Pipeline([
            StructuredPrune(sparsity=0.3),
            Distill(
                method="feature",
                feature_layers={"fc1": "fc1"},
                epochs=2,
                lr=1e-3,
                log_every=0,
            ),
        ])
        result = pipeline.run(
            model,
            training_data=train_loader,
            input_shape=(1, 32),
        )
        assert result.model is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDistillEdgeCases:
    def test_no_training_data_raises(self, mlp):
        stage = Distill(method="logit", epochs=1)
        ctx = PipelineContext(original_model=mlp)
        with pytest.raises(RuntimeError, match="training_data"):
            stage.apply(mlp, ctx)

    def test_alpha_zero_pure_task_loss(self, mlp, train_loader):
        """alpha=0 means no KD loss, only task loss."""
        stage = Distill(
            method="logit",
            alpha=0.0,
            task_loss_fn=nn.CrossEntropyLoss(),
            epochs=1,
            lr=1e-3,
            log_every=0,
        )
        ctx = PipelineContext(original_model=mlp, training_data=train_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_auto_match_no_common_layers_raises(self):
        class ModelA(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_a = nn.Linear(10, 10)

            def forward(self, x):
                return self.layer_a(x)

        class ModelB(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_b = nn.Linear(10, 10)

            def forward(self, x):
                return self.layer_b(x)

        with pytest.raises(RuntimeError, match="No common"):
            _auto_match_layers(ModelA(), ModelB())

    def test_fallback_to_calibration_data(self, mlp, train_loader):
        stage = Distill(method="logit", epochs=1, lr=1e-3, log_every=0)
        ctx = PipelineContext(original_model=mlp, calibration_data=train_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_teacher_none_uses_original(self, mlp, train_loader):
        stage = Distill(teacher=None, method="logit", epochs=1, lr=1e-3, log_every=0)
        ctx = PipelineContext(original_model=mlp, training_data=train_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# Recipe YAML round-trip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Precomputed targets
# ---------------------------------------------------------------------------


class TestPrecomputed:
    def test_precomputed_logit_distill(self, mlp, train_loader):
        """Precomputed mode: teacher logits supplied in data, no teacher model needed."""
        # First, generate teacher logits
        teacher = SmallMLP()
        teacher.eval()
        precomputed_batches = []
        with torch.no_grad():
            for batch in train_loader:
                x, y = batch
                t_logits = teacher(x)
                precomputed_batches.append((x, t_logits, y))

        pre_loader = DataLoader(
            TensorDataset(
                *[torch.cat(ts) for ts in zip(*[(b[0], b[1], b[2]) for b in precomputed_batches])]
            ),
            batch_size=16,
        )

        stage = Distill(method="logit", precomputed=True, epochs=2, lr=1e-3, log_every=0)
        ctx = PipelineContext(original_model=mlp, training_data=pre_loader)
        result = stage.apply(mlp, ctx)
        assert isinstance(result, nn.Module)

    def test_precomputed_feature_raises(self, mlp, train_loader):
        """Precomputed mode with feature matching should raise."""
        stage = Distill(method="feature", precomputed=True, epochs=1)
        ctx = PipelineContext(original_model=mlp, training_data=train_loader)
        with pytest.raises(ValueError, match="precomputed.*logit"):
            stage.apply(mlp, ctx)


# ---------------------------------------------------------------------------
# Real model smoke tests
# ---------------------------------------------------------------------------


class TestRealModelSmoke:
    """Minimal real-model distillation tests.

    These verify the pipeline works end-to-end on real architectures,
    not just toy MLPs. They don't need to converge — just run without errors.
    """

    @pytest.fixture
    def resnet_pair(self):
        """ResNet18 teacher/student pair with tiny input."""
        try:
            from torchvision.models import resnet18
        except ImportError:
            pytest.skip("torchvision not installed")
        torch.manual_seed(42)
        teacher = resnet18(num_classes=10)
        student = resnet18(num_classes=10)
        # Tiny dataset: 8 images, 3x32x32
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        return teacher, student, loader

    def test_resnet_logit_distill(self, resnet_pair):
        teacher, student, loader = resnet_pair
        stage = Distill(teacher=teacher, method="logit", epochs=1, lr=1e-3, log_every=0)
        ctx = PipelineContext(original_model=teacher, training_data=loader)
        result = stage.apply(student, ctx)
        # Verify it still produces output of correct shape
        result.eval()
        with torch.no_grad():
            out = result(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_resnet_feature_distill(self, resnet_pair):
        teacher, student, loader = resnet_pair
        stage = Distill(
            teacher=teacher,
            method="feature",
            feature_layers={"fc": "fc"},
            epochs=1,
            lr=1e-3,
            log_every=0,
        )
        ctx = PipelineContext(original_model=teacher, training_data=loader)
        result = stage.apply(student, ctx)
        assert isinstance(result, nn.Module)

    def test_resnet_prune_then_distill(self, resnet_pair):
        """Full pipeline: prune → feature distill on ResNet18."""
        from lobotomizer.stages.structured_prune import StructuredPrune

        teacher, student, loader = resnet_pair
        pipeline = Pipeline([
            StructuredPrune(sparsity=0.2),
            Distill(
                method="feature",
                feature_layers={"layer1.0.conv1": "layer1.0.conv1"},
                epochs=1,
                lr=1e-3,
                log_every=0,
            ),
        ])
        result = pipeline.run(student, training_data=loader, input_shape=(1, 3, 32, 32))
        assert result.model is not None


# ---------------------------------------------------------------------------
# Recipe YAML round-trip
# ---------------------------------------------------------------------------


class TestDistillRecipe:
    def test_yaml_round_trip(self, tmp_path):
        import yaml

        recipe = {
            "stages": [
                {"type": "distill", "method": "logit", "temperature": 4.0, "epochs": 3},
            ]
        }
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.dump(recipe))

        pipeline = build_pipeline_from_recipe(path)
        assert len(pipeline.stages) == 1
        assert isinstance(pipeline.stages[0], Distill)

    def test_yaml_with_prune_and_distill(self, tmp_path):
        import yaml

        recipe = {
            "stages": [
                {"type": "structured_prune", "sparsity": 0.2},
                {"type": "distill", "method": "logit", "epochs": 1},
            ]
        }
        path = tmp_path / "recipe.yaml"
        path.write_text(yaml.dump(recipe))

        pipeline = build_pipeline_from_recipe(path)
        assert len(pipeline.stages) == 2
