"""Quantization stage using torchao (replaces deprecated torch.ao.quantization)."""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
    quantize_,
)
from torchao.quantization.granularity import PerTensor

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)

# torchao uses config objects instead of dtype enums; we keep the "qint8"
# user-facing name for backward compatibility.
_SUPPORTED_DTYPES = {"qint8"}


class Quantize(Stage):
    """Quantize a model using torchao int8 quantization."""

    def __init__(self, method: str = "dynamic", dtype: str = "qint8") -> None:
        if method not in ("dynamic", "static"):
            raise ValueError(f"Unknown quantization method '{method}'. Choose 'dynamic' or 'static'.")
        if dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {sorted(_SUPPORTED_DTYPES)}")
        self._method = method
        self._dtype = dtype

    @property
    def name(self) -> str:
        return f"quantize({self._method}, {self._dtype})"

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        if self._method == "static":
            warnings.append(
                "Static quantization requires calibration_data in PipelineContext. "
                "Ensure it is provided."
            )
        try:
            p = next(model.parameters())
            if p.is_cuda:
                warnings.append(
                    "Model is on CUDA. torchao int8 quantization works best on CPU. "
                    "Model will be moved to CPU."
                )
        except StopIteration:
            pass
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        linear_params = sum(
            p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters()
        )
        if self._method == "dynamic":
            estimated_size_reduction = linear_params * 3 / 4  # 32bit -> 8bit saves 3/4
        else:
            estimated_size_reduction = total_params * 3 / 4
        return {
            "total_params": total_params,
            "estimated_bytes_saved": int(estimated_size_reduction * 4),  # 4 bytes per float32 param
        }

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        # Work on a copy — quantize_ is in-place
        model = copy.deepcopy(model).cpu()

        if self._method == "dynamic":
            quantize_(model, Int8DynamicActivationInt8WeightConfig())
            # Re-tie weights that deepcopy may have duplicated
            # (e.g. Whisper's proj_out ↔ decoder.embed_tokens)
            if hasattr(model, "tie_weights") and callable(model.tie_weights):
                model.tie_weights()
            return model

        # Static quantization
        if context.calibration_data is None:
            raise RuntimeError(
                "Static quantization requires calibration_data in PipelineContext."
            )

        model.eval()

        # Calibrate: collect max activation magnitude across all Linear layers
        max_abs = torch.tensor(0.0)

        def _calibration_hook(mod: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            nonlocal max_abs
            x = inp[0]
            max_abs = torch.max(max_abs, x.abs().max())

        hooks = []
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                hooks.append(mod.register_forward_hook(_calibration_hook))

        with torch.no_grad():
            for batch in context.calibration_data:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                model(x.cpu())

        for h in hooks:
            h.remove()

        act_scale = (max_abs / 127.0).reshape(1, 1)
        quantize_(
            model,
            Int8StaticActivationInt8WeightConfig(
                act_quant_scale=act_scale,
                granularity=PerTensor(),
            ),
        )
        return model
