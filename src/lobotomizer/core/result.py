"""Compression result container."""
from __future__ import annotations

import json
import pathlib
from typing import Any

import torch
import torch.nn as nn

from lobotomizer.stages.base import StageResult


class Result:
    """Container for compression results."""

    def __init__(
        self,
        original_model: nn.Module,
        compressed_model: nn.Module,
        stage_results: list[StageResult],
        profile_before: dict[str, Any],
        profile_after: dict[str, Any],
    ) -> None:
        self._original_model = original_model
        self._compressed_model = compressed_model
        self.stage_results = stage_results
        self.profile_before = profile_before
        self.profile_after = profile_after

    @property
    def model(self) -> nn.Module:
        """Return the compressed model."""
        return self._compressed_model

    def summary(self) -> str:
        """Return a formatted table comparing before/after."""
        rows: list[tuple[str, str, str, str]] = []
        for key in ("param_count", "param_count_trainable", "size_mb", "flops"):
            before = self.profile_before.get(key)
            after = self.profile_after.get(key)
            if before is None and after is None:
                continue
            b_str = _fmt(before)
            a_str = _fmt(after)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before:
                pct = (after - before) / before * 100
                delta = f"{pct:+.1f}%"
            else:
                delta = "—"
            rows.append((key, b_str, a_str, delta))

        if not rows:
            return "No metrics available."

        # Column widths
        headers = ("Metric", "Before", "After", "Δ")
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows))
            for i in range(4)
        ]

        def _row(vals: tuple[str, ...]) -> str:
            return "│ " + " │ ".join(v.ljust(w) for v, w in zip(vals, widths)) + " │"

        top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
        mid = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
        bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"

        lines = [top, _row(headers), mid]
        for r in rows:
            lines.append(_row(r))
        lines.append(bot)
        return "\n".join(lines)

    def sparsity_report(self) -> str:
        """Return a per-layer sparsity report for the compressed model.

        Shows the fraction of zero weights in each layer, useful for
        understanding what unstructured pruning actually did.
        """
        rows: list[tuple[str, str, str, str, str]] = []
        total_params = 0
        total_zeros = 0

        # Track modules with _weight_bias to avoid double-counting
        # (quantized modules expose weights on both parent and _packed_params child)
        modules_with_wb = {
            id(m) for m in self._compressed_model.modules()
            if hasattr(m, "_weight_bias")
        }

        for name, module in self._compressed_model.named_modules():
            # Get weight tensor — handle both regular and quantized modules
            weight = None
            if hasattr(module, "weight") and isinstance(getattr(module, "weight", None), torch.Tensor):
                weight = module.weight
            elif hasattr(module, "_weight_bias"):
                # Skip if a child also has _weight_bias (count from leaf only)
                has_child_wb = any(
                    id(child) in modules_with_wb
                    for child in module.children()
                )
                if has_child_wb:
                    continue
                try:
                    w, _ = module._weight_bias()
                    weight = w
                except (AttributeError, RuntimeError):
                    pass

            if weight is None:
                continue

            # Dequantize if needed (torchao quantized tensors don't support eq)
            try:
                numel = weight.numel()
                zeros = int((weight == 0).sum().item())
            except (NotImplementedError, RuntimeError):
                try:
                    w = weight.dequantize()
                    numel = w.numel()
                    zeros = int((w == 0).sum().item())
                except Exception:
                    numel = weight.numel()
                    zeros = 0
            total_params += numel
            total_zeros += zeros
            sparsity = zeros / numel if numel > 0 else 0.0
            shape = "×".join(str(s) for s in weight.shape)

            rows.append((
                name or "(root)",
                type(module).__name__,
                shape,
                f"{zeros:,}/{numel:,}",
                f"{sparsity:.1%}",
            ))

        if not rows:
            return "No weight tensors found."

        # Add total row
        total_sparsity = total_zeros / total_params if total_params > 0 else 0.0
        rows.append((
            "TOTAL", "", "",
            f"{total_zeros:,}/{total_params:,}",
            f"{total_sparsity:.1%}",
        ))

        headers = ("Layer", "Type", "Shape", "Zeros/Total", "Sparsity")
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows))
            for i in range(5)
        ]

        def _row(vals: tuple[str, ...], sep: str = "│") -> str:
            return f"{sep} " + f" {sep} ".join(v.ljust(w) for v, w in zip(vals, widths)) + f" {sep}"

        top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
        mid = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
        bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"

        lines = [top, _row(headers), mid]
        for i, r in enumerate(rows):
            if i == len(rows) - 1:  # separator before total
                lines.append(mid)
            lines.append(_row(r))
        lines.append(bot)
        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save compressed model and metadata to *path*."""
        p = pathlib.Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self._compressed_model.state_dict(), p / "model.pt")
        meta = {
            "profile_before": _serialisable(self.profile_before),
            "profile_after": _serialisable(self.profile_after),
            "stages": [sr.stage_name for sr in self.stage_results],
        }
        (p / "metadata.json").write_text(json.dumps(meta, indent=2))


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _serialisable(d: dict) -> dict:
    """Make a profile dict JSON-serialisable."""
    return {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in d.items()}
