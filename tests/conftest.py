"""Shared test fixtures."""
from __future__ import annotations

import pytest
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage


class IdentityStage(Stage):
    """A no-op stage for testing."""

    @property
    def name(self) -> str:
        return "identity"

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        return model


@pytest.fixture
def simple_model() -> nn.Module:
    return nn.Linear(32, 16)


@pytest.fixture
def identity_stage() -> IdentityStage:
    return IdentityStage()
