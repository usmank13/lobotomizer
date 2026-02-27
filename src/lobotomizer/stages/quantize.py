"""Quantization stage using PyTorch's native torch.ao.quantization."""
from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "qint8": torch.qint8,
}


class Quantize(Stage):
    """Quantize a model using PyTorch native quantization."""

    def __init__(self, method: str = "dynamic", dtype: str = "qint8") -> None:
        if method not in ("dynamic", "static"):
            raise ValueError(f"Unknown quantization method '{method}'. Choose 'dynamic' or 'static'.")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{dtype}'. Choose from: {list(_DTYPE_MAP)}")
        self._method = method
        self._dtype = dtype

    @property
    def name(self) -> str:
        return f"quantize({self._method}, {self._dtype})"

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        if self._method == "static":
            # We can't check context here, but warn about the requirement
            warnings.append(
                "Static quantization requires calibration_data in PipelineContext. "
                "Ensure it is provided."
            )
        # Check for CUDA params
        try:
            p = next(model.parameters())
            if p.is_cuda:
                warnings.append(
                    "Model is on CUDA. PyTorch quantization only works on CPU. "
                    "Model will be moved to CPU."
                )
        except StopIteration:
            pass
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        # Dynamic int8 roughly 4x smaller for targeted layers
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
        # Move to CPU if needed
        model = model.cpu()
        torch_dtype = _DTYPE_MAP[self._dtype]

        if self._method == "dynamic":
            quantized = torch.ao.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch_dtype,
            )
            return quantized

        # Static quantization
        if context.calibration_data is None:
            raise RuntimeError(
                "Static quantization requires calibration_data in PipelineContext."
            )

        model.eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # type: ignore[assignment]
        prepared = torch.ao.quantization.prepare(model)

        # Calibrate
        with torch.no_grad():
            for batch in context.calibration_data:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                prepared(x.cpu())

        quantized = torch.ao.quantization.convert(prepared)
        return quantized
