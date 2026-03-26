"""ONNX export for compressed models."""
from __future__ import annotations

import logging
import pathlib
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def to_onnx(
    model: nn.Module,
    output_path: str | pathlib.Path,
    input_shape: tuple[int, ...] | None = None,
    dummy_input: torch.Tensor | None = None,
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    device: str = "cpu",
) -> pathlib.Path:
    """Export a PyTorch model to ONNX format.

    Parameters
    ----------
    model : nn.Module
        The model to export (typically after compression).
    output_path : str or Path
        Where to save the .onnx file.
    input_shape : tuple, optional
        Shape of the dummy input tensor (e.g. ``(1, 3, 224, 224)``).
        Ignored if ``dummy_input`` is provided.
    dummy_input : Tensor, optional
        Explicit dummy input. If not provided, a random tensor of
        ``input_shape`` is created.
    opset_version : int
        ONNX opset version (default: 17).
    dynamic_axes : dict, optional
        Dynamic axes for variable-length dimensions (e.g. batch size).
        If not provided, defaults to making the first axis dynamic.
    input_names : sequence of str, optional
        Names for input tensors (default: ``["input"]``).
    output_names : sequence of str, optional
        Names for output tensors (default: ``["output"]``).
    device : str
        Device for the dummy input and model (default: "cpu").

    Returns
    -------
    pathlib.Path
        The path to the exported ONNX file.

    Raises
    ------
    ValueError
        If neither ``input_shape`` nor ``dummy_input`` is provided.
    """
    if dummy_input is None and input_shape is None:
        raise ValueError("Either input_shape or dummy_input must be provided.")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    model = model.to(device).eval()

    if dummy_input is None:
        dummy_input = torch.randn(*input_shape, device=device)
    else:
        dummy_input = dummy_input.to(device)

    if dynamic_axes is None:
        # Default: make batch dimension dynamic
        dynamic_axes = {
            input_names[0]: {0: "batch_size"},
            output_names[0]: {0: "batch_size"},
        }

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Exported ONNX model to %s (%.2f MB)", output_path, file_size_mb)
    return output_path


def validate_onnx(path: str | pathlib.Path) -> bool:
    """Check that an ONNX file is well-formed.

    Returns True if valid, raises on error. Requires ``onnx`` package.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "onnx package required for validation. Install with: pip install onnx"
        )

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    logger.info("ONNX model at %s is valid.", path)
    return True
