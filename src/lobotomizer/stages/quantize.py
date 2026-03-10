"""Quantization stage with pluggable method registry.

Supports:
- dynamic_int8: Dynamic int8 quantization via torchao (default, no extra deps)
- static_int8: Static int8 quantization via torchao (requires calibration data)
- int4_weight_only: INT4 weight-only quantization via torchao
- gptq: GPTQ quantization (requires auto-gptq)
- awq: AWQ quantization (requires autoawq)
"""
from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quantization method registry
# ---------------------------------------------------------------------------

class QuantMethod(ABC):
    """Base class for quantization methods."""

    name: str = ""
    requires_calibration: bool = False
    optional_deps: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    @abstractmethod
    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        """Apply quantization. Returns quantized model."""
        ...

    def validate(self, model: nn.Module) -> list[str]:
        """Return warnings for this method."""
        return []

    def estimate_compression_ratio(self) -> float:
        """Estimated size ratio (e.g., 0.25 = 4x smaller)."""
        return 0.5


_QUANT_METHODS: dict[str, type[QuantMethod]] = {}


def register_quant_method(name: str):
    """Decorator to register a quantization method."""
    def wrapper(cls: type[QuantMethod]) -> type[QuantMethod]:
        cls.name = name
        _QUANT_METHODS[name] = cls
        return cls
    return wrapper


def available_methods() -> list[str]:
    """Return list of registered quantization method names."""
    return sorted(_QUANT_METHODS.keys())


# ---------------------------------------------------------------------------
# Built-in methods
# ---------------------------------------------------------------------------

@register_quant_method("dynamic_int8")
class DynamicInt8(QuantMethod):
    """Dynamic int8 quantization via torchao."""

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        model = copy.deepcopy(model).cpu()
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        if hasattr(model, "tie_weights") and callable(model.tie_weights):
            model.tie_weights()
        return model

    def estimate_compression_ratio(self) -> float:
        return 0.25  # fp32 -> int8


@register_quant_method("static_int8")
class StaticInt8(QuantMethod):
    """Static int8 quantization via torchao (requires calibration data)."""

    requires_calibration = True

    def validate(self, model: nn.Module) -> list[str]:
        return [
            "Static quantization requires calibration_data in PipelineContext. "
            "Ensure it is provided."
        ]

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        from torchao.quantization import (
            Int8StaticActivationInt8WeightConfig,
            quantize_,
        )
        from torchao.quantization.granularity import PerTensor

        if context.calibration_data is None:
            raise RuntimeError(
                "Static quantization requires calibration_data in PipelineContext."
            )

        model = copy.deepcopy(model).cpu()
        model.eval()

        # Calibrate
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

    def estimate_compression_ratio(self) -> float:
        return 0.25


@register_quant_method("int4_weight_only")
class Int4WeightOnly(QuantMethod):
    """INT4 weight-only quantization via torchao."""

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        from torchao.quantization import Int4WeightOnlyConfig, quantize_

        model = copy.deepcopy(model).cpu()
        quantize_(model, Int4WeightOnlyConfig())
        if hasattr(model, "tie_weights") and callable(model.tie_weights):
            model.tie_weights()
        return model

    def validate(self, model: nn.Module) -> list[str]:
        warnings = []
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and mod.in_features % 32 != 0:
                warnings.append(
                    f"Layer '{name}' has in_features={mod.in_features} not divisible by 32. "
                    f"INT4 weight-only may skip this layer."
                )
        return warnings

    def estimate_compression_ratio(self) -> float:
        return 0.125  # fp32 -> int4


@register_quant_method("gptq")
class GPTQMethod(QuantMethod):
    """GPTQ quantization via auto-gptq."""

    requires_calibration = True
    optional_deps = ["auto_gptq"]

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            raise ImportError(
                "GPTQ quantization requires 'auto-gptq'. "
                "Install with: pip install auto-gptq"
            )

        bits = self.kwargs.get("bits", 4)
        group_size = self.kwargs.get("group_size", 128)
        desc_act = self.kwargs.get("desc_act", False)

        if context.calibration_data is None:
            raise RuntimeError(
                "GPTQ requires calibration_data in PipelineContext."
            )

        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
        )

        # auto-gptq works with HuggingFace models via from_pretrained.
        # For raw nn.Module, we wrap the quantize call.
        if hasattr(model, "config") and hasattr(model, "name_or_path"):
            # HuggingFace model path
            gptq_model = AutoGPTQForCausalLM.from_pretrained(
                model.name_or_path, quantize_config
            )
            # Collect calibration examples
            examples = []
            for batch in context.calibration_data:
                if isinstance(batch, dict):
                    examples.append(batch)
                elif isinstance(batch, (list, tuple)):
                    examples.append({"input_ids": batch[0]})
                else:
                    examples.append({"input_ids": batch})
            gptq_model.quantize(examples)
            return gptq_model.model
        else:
            raise TypeError(
                "GPTQ via auto-gptq requires a HuggingFace-style model with "
                "'config' and 'name_or_path' attributes. For raw nn.Module, "
                "consider using int4_weight_only instead."
            )

    def validate(self, model: nn.Module) -> list[str]:
        warnings = []
        try:
            import auto_gptq  # noqa: F401
        except ImportError:
            warnings.append(
                "auto-gptq is not installed. GPTQ will fail at apply time. "
                "Install with: pip install auto-gptq"
            )
        if not (hasattr(model, "config") and hasattr(model, "name_or_path")):
            warnings.append(
                "GPTQ via auto-gptq works best with HuggingFace models."
            )
        return warnings

    def estimate_compression_ratio(self) -> float:
        bits = self.kwargs.get("bits", 4)
        return bits / 32.0


@register_quant_method("awq")
class AWQMethod(QuantMethod):
    """AWQ quantization via autoawq."""

    requires_calibration = True
    optional_deps = ["awq"]

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError(
                "AWQ quantization requires 'autoawq'. "
                "Install with: pip install autoawq"
            )

        w_bit = self.kwargs.get("w_bit", 4)
        q_group_size = self.kwargs.get("q_group_size", 128)
        zero_point = self.kwargs.get("zero_point", True)

        if not (hasattr(model, "config") and hasattr(model, "name_or_path")):
            raise TypeError(
                "AWQ via autoawq requires a HuggingFace-style model with "
                "'config' and 'name_or_path' attributes."
            )

        if context.calibration_data is None:
            raise RuntimeError(
                "AWQ requires calibration_data in PipelineContext."
            )

        quant_config = {
            "w_bit": w_bit,
            "q_group_size": q_group_size,
            "zero_point": zero_point,
        }

        awq_model = AutoAWQForCausalLM.from_pretrained(model.name_or_path)

        # Collect calibration text
        calib_data = []
        for batch in context.calibration_data:
            if isinstance(batch, str):
                calib_data.append(batch)
            elif isinstance(batch, dict) and "text" in batch:
                calib_data.append(batch["text"])
            elif isinstance(batch, (list, tuple)):
                calib_data.extend(batch)

        tokenizer_name = self.kwargs.get("tokenizer", model.name_or_path)
        awq_model.quantize(calib_data, quant_config=quant_config, tokenizer=tokenizer_name)
        return awq_model.model

    def validate(self, model: nn.Module) -> list[str]:
        warnings = []
        try:
            import awq  # noqa: F401
        except ImportError:
            warnings.append(
                "autoawq is not installed. AWQ will fail at apply time. "
                "Install with: pip install autoawq"
            )
        if not (hasattr(model, "config") and hasattr(model, "name_or_path")):
            warnings.append(
                "AWQ via autoawq works best with HuggingFace models."
            )
        return warnings

    def estimate_compression_ratio(self) -> float:
        w_bit = self.kwargs.get("w_bit", 4)
        return w_bit / 32.0


# ---------------------------------------------------------------------------
# Backward-compatible method aliases
# ---------------------------------------------------------------------------
_METHOD_ALIASES = {
    "dynamic": "dynamic_int8",
    "static": "static_int8",
}

# Backward-compatible dtype set
_SUPPORTED_DTYPES = {"qint8", "int4", "auto"}


# ---------------------------------------------------------------------------
# Main Stage class
# ---------------------------------------------------------------------------

class Quantize(Stage):
    """Quantize a model using pluggable quantization methods.

    Parameters
    ----------
    method : str
        Quantization method. One of: dynamic, static, dynamic_int8, static_int8,
        int4_weight_only, gptq, awq.  Default: "dynamic".
    dtype : str
        Kept for backward compatibility. Ignored for new methods.
    **kwargs
        Extra keyword arguments passed to the quantization method
        (e.g., bits=4, group_size=128 for GPTQ).
    """

    def __init__(self, method: str = "dynamic", dtype: str = "qint8", **kwargs: Any) -> None:
        # Resolve aliases
        resolved = _METHOD_ALIASES.get(method, method)
        if resolved not in _QUANT_METHODS:
            raise ValueError(
                f"Unknown quantization method '{method}'. "
                f"Available: {available_methods()} "
                f"(aliases: {list(_METHOD_ALIASES.keys())})"
            )
        self._method_name = resolved
        self._dtype = dtype
        self._kwargs = kwargs
        self._quant_method = _QUANT_METHODS[resolved](**kwargs)

    @property
    def name(self) -> str:
        return f"quantize({self._method_name})"

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        warnings.extend(self._quant_method.validate(model))
        try:
            p = next(model.parameters())
            if p.is_cuda:
                warnings.append(
                    "Model is on CUDA. Some quantization methods work best on CPU. "
                    "Model may be moved to CPU."
                )
        except StopIteration:
            pass
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        linear_params = sum(
            p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters()
        )
        ratio = self._quant_method.estimate_compression_ratio()
        estimated_bytes_saved = int(linear_params * (1 - ratio) * 4)
        return {
            "total_params": total_params,
            "estimated_bytes_saved": estimated_bytes_saved,
            "compression_ratio": ratio,
            "method": self._method_name,
        }

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        logger.info("Applying quantization method: %s", self._method_name)
        return self._quant_method.apply(model, context)
