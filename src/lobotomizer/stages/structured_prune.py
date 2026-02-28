"""Structured pruning that physically removes rows/columns from Linear layers."""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)


class StructuredPrune(Stage):
    """Remove entire neurons (rows/columns) from Linear layers.

    Unlike unstructured pruning which just zeros weights, this stage
    physically shrinks weight matrices, changing the architecture and
    producing real speedups.

    Parameters
    ----------
    sparsity : float
        Fraction of output neurons to remove per layer (0.0 to 1.0).
    criterion : "l1" or "l2"
        Norm used to rank neurons for removal.  L1 = sum of absolute
        weights per output neuron; L2 = Euclidean norm.
    """

    def __init__(
        self,
        sparsity: float = 0.3,
        criterion: Literal["l1", "l2"] = "l1",
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0, 1), got {sparsity}")
        if criterion not in ("l1", "l2"):
            raise ValueError(f"Criterion must be 'l1' or 'l2', got '{criterion}'")
        self._sparsity = sparsity
        self._criterion = criterion

    @property
    def name(self) -> str:
        return f"structured_prune({self._criterion}, {self._sparsity})"

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if not linears:
            warnings.append("Model has no Linear layers for structured pruning.")
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        prunable = sum(
            p.numel() for m in model.modules() if isinstance(m, nn.Linear)
            for p in m.parameters()
        )
        estimated_removed = int(prunable * self._sparsity)
        return {
            "total_params": total_params,
            "prunable_params": prunable,
            "estimated_removed": estimated_removed,
            "estimated_remaining": total_params - estimated_removed,
        }

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        linear_pairs = _find_linear_chains(model)
        if not linear_pairs and not any(isinstance(m, nn.Linear) for m in model.modules()):
            logger.warning("No Linear layers found; returning model unchanged.")
            return model

        pruned_outputs: dict[nn.Linear, torch.Tensor] = {}

        # First pass: decide which output indices to keep per layer
        for layer in _all_linears(model):
            keep = self._compute_keep_indices(layer)
            pruned_outputs[layer] = keep

        # Build a mapping from layer to its downstream partner (if any)
        downstream: dict[nn.Linear, nn.Linear] = {}
        for src, dst in linear_pairs:
            downstream[src] = dst

        # Second pass: apply the pruning
        for layer in _all_linears(model):
            keep = pruned_outputs[layer]
            _prune_output_dim(layer, keep)

            # Propagate to downstream layer's input dimension
            if layer in downstream:
                _prune_input_dim(downstream[layer], keep)

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_keep_indices(self, layer: nn.Linear) -> torch.Tensor:
        """Return sorted indices of output neurons to keep."""
        weight = layer.weight.data  # (out_features, in_features)
        n_out = weight.shape[0]
        n_remove = max(1, int(n_out * self._sparsity))
        n_keep = n_out - n_remove
        if n_keep < 1:
            n_keep = 1
            n_remove = n_out - 1

        if self._criterion == "l1":
            scores = weight.abs().sum(dim=1)
        else:  # l2
            scores = weight.norm(p=2, dim=1)

        _, indices = scores.sort(descending=True)
        keep = indices[:n_keep].sort().values
        return keep


def _prune_output_dim(layer: nn.Linear, keep: torch.Tensor) -> None:
    """Remove rows from weight (and bias) — shrinks out_features."""
    layer.weight = nn.Parameter(layer.weight.data[keep])
    if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.data[keep])
    layer.out_features = layer.weight.shape[0]


def _prune_input_dim(layer: nn.Linear, keep: torch.Tensor) -> None:
    """Remove columns from weight — shrinks in_features."""
    layer.weight = nn.Parameter(layer.weight.data[:, keep])
    layer.in_features = layer.weight.shape[1]


def _all_linears(model: nn.Module) -> list[nn.Linear]:
    """Return all Linear layers in forward order."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _find_linear_chains(model: nn.Module) -> list[tuple[nn.Linear, nn.Linear]]:
    """Find consecutive Linear layer pairs (possibly separated by non-parametric layers).

    Walks named children of Sequential-like containers and pairs up
    Linear layers that feed into each other (with only activation /
    dropout / normalization in between).
    """
    pairs: list[tuple[nn.Linear, nn.Linear]] = []

    # For Sequential-style models, walk children in order
    _find_pairs_recursive(model, pairs)
    return pairs


_PASSTHROUGH_TYPES = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
    nn.Dropout, nn.BatchNorm1d, nn.LayerNorm, nn.Identity,
)


def _find_pairs_recursive(
    module: nn.Module,
    pairs: list[tuple[nn.Linear, nn.Linear]],
) -> None:
    """Recursively find Linear→Linear pairs in Sequential-like containers."""
    children = list(module.children())
    if not children:
        return

    # Collect linear layers at this level (skipping passthrough layers)
    linears_at_level: list[nn.Linear] = []
    for child in children:
        if isinstance(child, nn.Linear):
            linears_at_level.append(child)
        elif isinstance(child, _PASSTHROUGH_TYPES):
            continue  # skip, doesn't break the chain
        else:
            # Non-passthrough, non-linear: break the chain and recurse
            if len(linears_at_level) >= 2:
                for i in range(len(linears_at_level) - 1):
                    pairs.append((linears_at_level[i], linears_at_level[i + 1]))
            linears_at_level = []
            _find_pairs_recursive(child, pairs)

    # Final flush
    if len(linears_at_level) >= 2:
        for i in range(len(linears_at_level) - 1):
            pairs.append((linears_at_level[i], linears_at_level[i + 1]))

    # Also recurse into all children that aren't simple layers
    for child in children:
        if not isinstance(child, (nn.Linear, *_PASSTHROUGH_TYPES)):
            pass  # already recursed above
        # Don't recurse into Linear or passthrough
