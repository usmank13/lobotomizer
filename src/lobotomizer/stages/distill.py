"""Knowledge distillation stage."""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lobotomizer.stages.adapters import FeatureAdapter, resolve_adapter
from lobotomizer.stages.base import PipelineContext, Stage
from lobotomizer.stages.distill_losses import (
    CombinedKDLoss,
    FeatureMatchingLoss,
    KDLoss,
)
from lobotomizer.stages.hooks import ForwardHookManager

logger = logging.getLogger(__name__)


def _auto_match_layers(student: nn.Module, teacher: nn.Module) -> dict[str, str]:
    """Find common nn.Linear/nn.Conv2d layer names between student and teacher."""
    target_types = (nn.Linear, nn.Conv2d)

    def _get_names(model: nn.Module) -> set[str]:
        return {
            name
            for name, mod in model.named_modules()
            if isinstance(mod, target_types)
        }

    common = _get_names(student) & _get_names(teacher)
    if not common:
        raise RuntimeError(
            "No common Linear/Conv2d layers found between student and teacher. "
            "Provide explicit feature_layers mapping."
        )
    mapping = {name: name for name in sorted(common)}
    logger.info("Auto-matched %d feature layers: %s", len(mapping), list(mapping.keys()))
    return mapping


def _get_feature_dim(model: nn.Module, layer_name: str) -> int:
    """Infer output dimension of a named layer."""
    mod = dict(model.named_modules())[layer_name]
    if isinstance(mod, nn.Linear):
        return mod.out_features
    if isinstance(mod, nn.Conv2d):
        return mod.out_channels
    raise ValueError(f"Cannot infer feature dim for {type(mod)}")


class Distill(Stage):
    """Knowledge distillation stage.

    Trains the student model to mimic a teacher using logit-based KD,
    feature matching, or both.

    Parameters
    ----------
    teacher : nn.Module | str | None
        Teacher model. ``None`` uses ``context.original_model``.
        A string is interpreted as a path to ``torch.load()``.
    method : str
        ``"logit"``, ``"feature"``, or ``"both"``.
    temperature : float
        Softmax temperature for logit KD.
    alpha : float
        Weight for KD loss (``1 - alpha`` goes to task loss).
    feature_layers : dict[str, str] | None
        Student → teacher layer name mapping. ``None`` triggers auto-matching.
    adapter
        Adapter specification (see :func:`resolve_adapter`).
    task_loss_fn : callable | None
        Task-specific loss function.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    optimizer_cls : type | None
        Optimizer class (default: ``torch.optim.Adam``).
    optimizer_kwargs : dict | None
        Extra kwargs for the optimizer.
    scheduler_cls : type | None
        LR scheduler class.
    scheduler_kwargs : dict | None
        Extra kwargs for the scheduler.
    precomputed : bool
        If ``True``, training data batches are expected to contain
        ``(inputs, teacher_logits)`` (or ``(inputs, teacher_logits, labels)``
        for task loss). The teacher model is not loaded or run, avoiding
        the memory cost of holding two models simultaneously.
    log_every : int
        Log training stats every N batches.
    """

    def __init__(
        self,
        teacher: nn.Module | str | None = None,
        method: str = "logit",
        temperature: float = 4.0,
        alpha: float = 1.0,
        feature_layers: dict[str, str] | None = None,
        adapter: Any = None,
        task_loss_fn: Callable | None = None,
        epochs: int = 5,
        lr: float = 1e-4,
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        scheduler_cls: type | None = None,
        scheduler_kwargs: dict | None = None,
        precomputed: bool = False,
        log_every: int = 50,
    ) -> None:
        if method not in ("logit", "feature", "both"):
            raise ValueError(f"method must be 'logit', 'feature', or 'both', got '{method}'")
        self._teacher = teacher
        self._method = method
        self._temperature = temperature
        self._alpha = alpha
        self._feature_layers = feature_layers
        self._adapter = adapter
        self._task_loss_fn = task_loss_fn
        self._epochs = epochs
        self._lr = lr
        self._optimizer_cls = optimizer_cls or torch.optim.AdamW
        self._optimizer_kwargs = optimizer_kwargs or {}
        self._scheduler_cls = scheduler_cls
        self._scheduler_kwargs = scheduler_kwargs or {}
        self._precomputed = precomputed
        self._log_every = log_every

    @property
    def name(self) -> str:
        return f"distill({self._method}, T={self._temperature})"

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        if self._method in ("feature", "both") and self._feature_layers is None:
            warnings.append(
                "No feature_layers specified; will attempt auto-matching at apply time."
            )
        return warnings

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        # 1. Resolve teacher (skip in precomputed mode)
        teacher: nn.Module | None = None
        if not self._precomputed:
            teacher = self._resolve_teacher(context)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher = teacher.to(context.device)
        elif self._method in ("feature", "both"):
            raise ValueError(
                "precomputed=True only supports method='logit'. "
                "Feature matching requires a live teacher for hook-based capture."
            )

        # 2. Resolve training data
        data = self._resolve_data(context)

        # 3. Feature setup (only when not precomputed)
        student_hooks = ForwardHookManager()
        teacher_hooks = ForwardHookManager()
        adapters: dict[str, nn.Module] = {}
        feature_loss: FeatureMatchingLoss | None = None
        layer_mapping: dict[str, str] | None = None

        use_features = self._method in ("feature", "both") and not self._precomputed
        if use_features:
            assert teacher is not None
            layer_mapping = self._feature_layers or _auto_match_layers(model, teacher)
            student_layers = list(layer_mapping.keys())
            teacher_layers = list(layer_mapping.values())
            student_hooks.register(model, student_layers)
            teacher_hooks.register(teacher, teacher_layers)

            for s_layer, t_layer in layer_mapping.items():
                s_dim = _get_feature_dim(model, s_layer)
                t_dim = _get_feature_dim(teacher, t_layer)
                adapters[s_layer] = resolve_adapter(self._adapter, s_dim, t_dim).to(context.device)

            feature_loss = FeatureMatchingLoss(adapters)

        # 4. Build loss
        kd_loss = KDLoss(self._temperature) if self._method in ("logit", "both") else None
        combined_loss = CombinedKDLoss(
            alpha=self._alpha,
            beta=1.0,
            kd_loss=kd_loss,
            feature_loss=feature_loss,
            task_loss_fn=self._task_loss_fn,
        )

        # 5. Optimizer
        params = list(model.parameters())
        for a in adapters.values():
            params.extend(a.parameters())
        optimizer = self._optimizer_cls(params, lr=self._lr, **self._optimizer_kwargs)

        scheduler = None
        if self._scheduler_cls is not None:
            scheduler = self._scheduler_cls(optimizer, **self._scheduler_kwargs)

        # 6. Training loop
        model.train()
        device = context.device
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_idx, batch in enumerate(data):
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                    if self._precomputed:
                        # batch = (inputs, teacher_logits[, labels])
                        teacher_out = batch[1].to(device)
                        targets = batch[2].to(device) if len(batch) > 2 else None
                    else:
                        targets = batch[1].to(device) if len(batch) > 1 else None
                else:
                    x = batch.to(device)
                    targets = None

                student_out = model(x)
                if not self._precomputed:
                    with torch.no_grad():
                        assert teacher is not None
                        teacher_out = teacher(x)

                s_feats = student_hooks.pop_features() if use_features else None
                t_feats = teacher_hooks.pop_features() if use_features else None

                loss = combined_loss(
                    student_out,
                    teacher_out,
                    student_features=s_feats,
                    teacher_features=t_feats,
                    targets=targets,
                )

                optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                if self._log_every and (batch_idx + 1) % self._log_every == 0:
                    logger.info(
                        "Epoch %d batch %d loss=%.6f",
                        epoch + 1,
                        batch_idx + 1,
                        loss.item(),
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info("Epoch %d/%d avg_loss=%.6f", epoch + 1, self._epochs, avg_loss)

            if scheduler is not None:
                scheduler.step()

        # 7. Cleanup
        student_hooks.remove_hooks()
        teacher_hooks.remove_hooks()

        model.eval()
        return model

    def _resolve_teacher(self, context: PipelineContext) -> nn.Module:
        """Resolve teacher model from config or context."""
        if self._teacher is None:
            import copy
            return copy.deepcopy(context.original_model)
        if isinstance(self._teacher, nn.Module):
            return self._teacher
        if isinstance(self._teacher, str):
            return torch.load(self._teacher, weights_only=False)
        raise TypeError(f"Cannot resolve teacher of type {type(self._teacher)}")

    def _resolve_data(self, context: PipelineContext) -> DataLoader:
        """Get training data from context."""
        training_data = getattr(context, "training_data", None)
        if training_data is not None:
            return training_data
        if context.calibration_data is not None:
            logger.warning("No training_data; falling back to calibration_data.")
            return context.calibration_data
        raise RuntimeError(
            "Distillation requires training_data or calibration_data in PipelineContext."
        )
