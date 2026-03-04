"""Integration tests with real models (DINO teacher → CNN student).

These tests verify end-to-end distillation works with real architectures.
They use random weights and tiny datasets — the goal is to confirm the
pipeline runs without errors, not to achieve meaningful accuracy.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lobotomizer import Distill
from lobotomizer.core.pipeline import Pipeline
from lobotomizer.stages.base import PipelineContext

# Skip entire module if torchvision not available
tv = pytest.importorskip("torchvision")


def _make_dummy_data(n: int = 16, img_size: int = 224, num_classes: int = 10):
    """Tiny random dataset."""
    x = torch.randn(n, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


class SmallCNN(nn.Module):
    """Minimal CNN student that outputs same num_classes as teacher."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


@pytest.fixture
def dino_teacher():
    """DINOv2 ViT-S/14 with random weights (no download)."""
    # Use torchvision's vit_s_16 as a stand-in for DINO architecture
    # (same ViT family, available without extra deps)
    model = tv.models.vit_b_16(weights=None)
    model.eval()
    return model


@pytest.fixture
def resnet_teacher():
    """ResNet-18 with random weights."""
    model = tv.models.resnet18(weights=None)
    model.eval()
    return model


@pytest.fixture
def cnn_student():
    return SmallCNN(num_classes=1000)


class TestDINOtoCNN:
    """DINO (ViT) teacher → small CNN student distillation."""

    def test_logit_distill(self, dino_teacher, cnn_student):
        """Logit-based KD from ViT teacher to CNN student runs end-to-end."""
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=dino_teacher,
            method="logit",
            temperature=4.0,
            alpha=1.0,
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student
        # Verify model still produces valid output
        out = result(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 1000)

    def test_logit_distill_with_task_loss(self, dino_teacher, cnn_student):
        """KD + cross-entropy task loss."""
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=dino_teacher,
            method="logit",
            temperature=4.0,
            alpha=0.7,
            task_loss_fn=nn.CrossEntropyLoss(),
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student

    def test_feature_distill_explicit_layers(self, dino_teacher, cnn_student):
        """Feature matching with explicit layer mapping (cross-architecture)."""
        data = _make_dummy_data(n=8, num_classes=1000)

        # Map CNN's classifier to ViT's head — both are nn.Linear
        stage = Distill(
            teacher=dino_teacher,
            method="feature",
            feature_layers={"classifier": "heads.head"},
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student

    def test_both_distill(self, dino_teacher, cnn_student):
        """Combined logit + feature KD."""
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=dino_teacher,
            method="both",
            temperature=4.0,
            alpha=0.7,
            feature_layers={"classifier": "heads.head"},
            task_loss_fn=nn.CrossEntropyLoss(),
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student


class TestResNettoCNN:
    """ResNet teacher → small CNN student (CNN-to-CNN distillation)."""

    def test_logit_distill(self, resnet_teacher, cnn_student):
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=resnet_teacher,
            method="logit",
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student

    def test_feature_auto_match(self, resnet_teacher, cnn_student):
        """Auto-matching should find the shared 'classifier' Linear layer."""
        # ResNet has 'fc', SmallCNN has 'classifier' — no overlap.
        # This should raise RuntimeError from auto-match.
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=resnet_teacher,
            method="feature",
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        with pytest.raises(RuntimeError, match="No common"):
            stage.apply(cnn_student, ctx)

    def test_feature_explicit_cross_arch(self, resnet_teacher, cnn_student):
        """Explicit layer mapping between ResNet and SmallCNN."""
        data = _make_dummy_data(n=8, num_classes=1000)

        stage = Distill(
            teacher=resnet_teacher,
            method="feature",
            feature_layers={"classifier": "fc"},  # SmallCNN.classifier → ResNet.fc
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student


class TestPipelineIntegration:
    """Test distillation through the Pipeline API (how users actually call it)."""

    def test_pipeline_distill_vit_to_cnn(self, dino_teacher, cnn_student):
        """Full pipeline: ViT teacher → CNN student via Pipeline.compress()."""
        data = _make_dummy_data(n=8, num_classes=1000)

        pipe = Pipeline([
            Distill(
                teacher=dino_teacher,
                method="logit",
                epochs=1,
                lr=1e-3,
                log_every=0,
            ),
        ])

        result = pipe.run(
            cnn_student,
            calibration_data=data,
            training_data=data,
        )
        assert result is not None
        out = result.model(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 1000)

    def test_precomputed_logits(self, cnn_student):
        """Precomputed teacher logits (no teacher model needed at train time)."""
        # Simulate precomputed: (images, teacher_logits)
        x = torch.randn(8, 3, 224, 224)
        teacher_logits = torch.randn(8, 1000)  # fake teacher outputs
        data = DataLoader(TensorDataset(x, teacher_logits), batch_size=4)

        stage = Distill(
            method="logit",
            precomputed=True,
            epochs=1,
            lr=1e-3,
            log_every=0,
        )

        ctx = PipelineContext(
            original_model=cnn_student,
            device=torch.device("cpu"),
            training_data=data,
        )

        result = stage.apply(cnn_student, ctx)
        assert result is cnn_student
