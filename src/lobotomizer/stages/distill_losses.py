"""Loss functions for knowledge distillation."""
from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class KDLoss(nn.Module):
    """Hinton-style knowledge distillation loss.

    Computes ``KL(softmax(student/T) || softmax(teacher/T)) * T²``.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        s = F.log_softmax(student_logits / T, dim=-1)
        t = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(s, t, reduction="batchmean") * (T * T)


class FeatureMatchingLoss(nn.Module):
    """MSE loss between adapted student and teacher intermediate features."""

    def __init__(self, adapters: dict[str, nn.Module]) -> None:
        super().__init__()
        # Use nn.ModuleList + a name→index map because nn.ModuleDict
        # doesn't allow dots in keys (common in nested layer names).
        self._adapter_names = list(adapters.keys())
        self._adapter_list = nn.ModuleList(adapters[k] for k in self._adapter_names)
        self._name_to_idx = {n: i for i, n in enumerate(self._adapter_names)}

    def forward(
        self,
        student_features: dict[str, torch.Tensor],
        teacher_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(student_features.values())).device)
        for student_layer, teacher_layer in self._layer_pairs(student_features, teacher_features):
            s_feat = student_features[student_layer]
            t_feat = teacher_features[teacher_layer]
            # Adapter key is the student layer name
            adapted = self._adapter_list[self._name_to_idx[student_layer]](s_feat)
            loss = loss + F.mse_loss(adapted, t_feat)
        return loss

    def _layer_pairs(
        self,
        student_features: dict[str, torch.Tensor],
        teacher_features: dict[str, torch.Tensor],
    ) -> list[tuple[str, str]]:
        """Pair student and teacher layers by order."""
        s_keys = list(student_features.keys())
        t_keys = list(teacher_features.keys())
        return list(zip(s_keys, t_keys))


class CombinedKDLoss(nn.Module):
    """Weighted combination of KD loss, feature matching loss, and task loss.

    ``total = alpha * kd + beta * feature + (1 - alpha) * task``
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        kd_loss: KDLoss | None,
        feature_loss: FeatureMatchingLoss | None,
        task_loss_fn: Callable | None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kd_loss = kd_loss
        self.feature_loss = feature_loss
        self.task_loss_fn = task_loss_fn

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: dict[str, torch.Tensor] | None = None,
        teacher_features: dict[str, torch.Tensor] | None = None,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = student_logits.device
        total = torch.tensor(0.0, device=device)

        if self.kd_loss is not None:
            total = total + self.alpha * self.kd_loss(student_logits, teacher_logits)

        if self.feature_loss is not None and student_features and teacher_features:
            total = total + self.beta * self.feature_loss(student_features, teacher_features)

        if self.task_loss_fn is not None and targets is not None:
            total = total + (1.0 - self.alpha) * self.task_loss_fn(student_logits, targets)

        return total
