"""Structured pruning that physically removes rows/columns from Linear and Conv2d layers."""
from __future__ import annotations

import logging
from typing import Literal, Union

import torch
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)

# Layer types that structured pruning can handle
_PRUNABLE = (nn.Linear, nn.Conv2d)
PrunableLayer = Union[nn.Linear, nn.Conv2d]

# Layers that don't break a chain between prunable layers
_PASSTHROUGH_TYPES = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
    nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
    nn.Identity, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
)


class StructuredPrune(Stage):
    """Remove entire neurons/filters from Linear and Conv2d layers.

    Unlike unstructured pruning which just zeros weights, this stage
    physically shrinks weight matrices, changing the architecture and
    producing real speedups.

    For Conv2d layers, this removes entire output filters (along dim 0
    of the weight tensor) and propagates the change to downstream layers'
    input channels and any intervening BatchNorm2d layers.

    Parameters
    ----------
    sparsity : float
        Fraction of output neurons/filters to remove per layer (0.0 to 1.0).
    criterion : "l1" or "l2"
        Norm used to rank neurons/filters for removal.
    protect_output : bool
        If ``True`` (default), automatically excludes output layers —
        layers with no downstream consumer — from pruning.
    exclude_layers : set[str] | list[str] | None
        Named layers to exclude from pruning entirely.
    """

    def __init__(
        self,
        sparsity: float = 0.3,
        criterion: Literal["l1", "l2"] = "l1",
        protect_output: bool = True,
        exclude_layers: set[str] | list[str] | None = None,
    ) -> None:
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"Sparsity must be in [0, 1), got {sparsity}")
        if criterion not in ("l1", "l2"):
            raise ValueError(f"Criterion must be 'l1' or 'l2', got '{criterion}'")
        self._sparsity = sparsity
        self._criterion = criterion
        self._protect_output = protect_output
        self._exclude_layers: set[str] = set(exclude_layers) if exclude_layers else set()

    @property
    def name(self) -> str:
        return f"structured_prune({self._criterion}, {self._sparsity})"

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        has_prunable = any(isinstance(m, _PRUNABLE) for m in model.modules())
        if not has_prunable:
            warnings.append("Model has no Linear or Conv2d layers for structured pruning.")
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        prunable = sum(
            p.numel() for m in model.modules() if isinstance(m, _PRUNABLE)
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
        chains = _find_prunable_chains(model)
        if not chains and not any(isinstance(m, _PRUNABLE) for m in model.modules()):
            logger.warning("No prunable layers found; returning model unchanged.")
            return model

        # Build module → name mapping
        module_to_name: dict[nn.Module, str] = {
            mod: name for name, mod in model.named_modules()
        }

        # downstream: layer → (next_prunable_layer, [intermediate_batchnorms])
        downstream: dict[PrunableLayer, tuple[PrunableLayer, list[nn.Module]]] = {}
        for src, dst, intermediates in chains:
            downstream[src] = (dst, intermediates)

        # Determine which layers to skip
        skip: set[PrunableLayer] = set()

        # User-specified exclusions
        if self._exclude_layers:
            named = dict(model.named_modules())
            for layer_name in self._exclude_layers:
                if layer_name in named and isinstance(named[layer_name], _PRUNABLE):
                    skip.add(named[layer_name])
                    logger.info("Excluding layer '%s' from pruning (user-specified).", layer_name)
                elif layer_name not in named:
                    logger.warning("Excluded layer '%s' not found in model.", layer_name)

        # Auto-protect output layers (no downstream consumer)
        if self._protect_output:
            all_prunable = _all_prunable(model)
            for layer in all_prunable:
                if layer not in downstream:
                    layer_name = module_to_name.get(layer, "<unknown>")
                    skip.add(layer)
                    logger.info(
                        "Protecting output layer '%s' from pruning (no downstream consumer).",
                        layer_name,
                    )

        # First pass: compute keep indices
        pruned_outputs: dict[PrunableLayer, torch.Tensor] = {}
        for layer in _all_prunable(model):
            if layer in skip:
                continue
            keep = self._compute_keep_indices(layer)
            pruned_outputs[layer] = keep

        # Second pass: apply pruning
        for layer in _all_prunable(model):
            if layer not in pruned_outputs:
                continue
            keep = pruned_outputs[layer]
            _prune_output_dim(layer, keep)

            # Propagate to downstream layer's input dimension + intermediates
            if layer in downstream:
                next_layer, intermediates = downstream[layer]
                for inter in intermediates:
                    _prune_batchnorm(inter, keep)
                _prune_input_dim(next_layer, keep)

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_keep_indices(self, layer: PrunableLayer) -> torch.Tensor:
        """Return sorted indices of output neurons/filters to keep."""
        weight = layer.weight.data
        n_out = weight.shape[0]
        n_remove = max(1, int(n_out * self._sparsity))
        n_keep = n_out - n_remove
        if n_keep < 1:
            n_keep = 1
            n_remove = n_out - 1

        if isinstance(layer, nn.Linear):
            # (out_features, in_features)
            if self._criterion == "l1":
                scores = weight.abs().sum(dim=1)
            else:
                scores = weight.norm(p=2, dim=1)
        else:
            # Conv2d: (out_channels, in_channels, kH, kW) → flatten per filter
            flat = weight.view(n_out, -1)
            if self._criterion == "l1":
                scores = flat.abs().sum(dim=1)
            else:
                scores = flat.norm(p=2, dim=1)

        _, indices = scores.sort(descending=True)
        keep = indices[:n_keep].sort().values
        return keep


# ======================================================================
# Pruning operations
# ======================================================================

def _prune_output_dim(layer: PrunableLayer, keep: torch.Tensor) -> None:
    """Remove output neurons/filters — shrinks out_features/out_channels."""
    layer.weight = nn.Parameter(layer.weight.data[keep])
    if layer.bias is not None:
        layer.bias = nn.Parameter(layer.bias.data[keep])

    if isinstance(layer, nn.Linear):
        layer.out_features = layer.weight.shape[0]
    else:
        layer.out_channels = layer.weight.shape[0]


def _prune_input_dim(layer: PrunableLayer, keep: torch.Tensor) -> None:
    """Remove input neurons/channels — shrinks in_features/in_channels."""
    if isinstance(layer, nn.Linear):
        layer.weight = nn.Parameter(layer.weight.data[:, keep])
        layer.in_features = layer.weight.shape[1]
    else:
        # Conv2d: dim 1 is in_channels
        layer.weight = nn.Parameter(layer.weight.data[:, keep])
        layer.in_channels = layer.weight.shape[1]


def _prune_batchnorm(bn: nn.Module, keep: torch.Tensor) -> None:
    """Shrink a BatchNorm layer to match pruned channel count."""
    if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return
    bn.num_features = keep.shape[0]
    if bn.weight is not None:
        bn.weight = nn.Parameter(bn.weight.data[keep])
    if bn.bias is not None:
        bn.bias = nn.Parameter(bn.bias.data[keep])
    if bn.running_mean is not None:
        bn.running_mean = bn.running_mean[keep]
    if bn.running_var is not None:
        bn.running_var = bn.running_var[keep]


# ======================================================================
# Chain finding
# ======================================================================

def _all_prunable(model: nn.Module) -> list[PrunableLayer]:
    """Return all prunable layers in forward order."""
    return [m for m in model.modules() if isinstance(m, _PRUNABLE)]


def _find_prunable_chains(
    model: nn.Module,
) -> list[tuple[PrunableLayer, PrunableLayer, list[nn.Module]]]:
    """Find consecutive prunable layer pairs, possibly separated by passthrough layers.

    Returns list of (source, destination, intermediates) tuples where
    intermediates are BatchNorm layers between src and dst that need
    to be adjusted when src is pruned.
    """
    chains: list[tuple[PrunableLayer, PrunableLayer, list[nn.Module]]] = []
    _find_chains_recursive(model, chains)
    return chains


def _find_chains_recursive(
    module: nn.Module,
    chains: list[tuple[PrunableLayer, PrunableLayer, list[nn.Module]]],
) -> None:
    """Recursively find prunable→prunable pairs in Sequential-like containers.

    Only forms chains within ``nn.Sequential`` containers to avoid breaking
    skip/residual connections in modules like ResNet BasicBlock.
    """
    children = list(module.children())
    if not children:
        return

    is_sequential = isinstance(module, nn.Sequential)

    # Track current chain of prunable layers and intermediates at this level
    prunable_at_level: list[PrunableLayer] = []
    intermediates_at_level: list[list[nn.Module]] = []
    current_intermediates: list[nn.Module] = []

    # In non-Sequential modules, only chain Linear layers (Conv2d may have
    # skip connections we can't detect from structure alone). Sequential
    # containers are safe to chain all prunable types.
    allow_conv = is_sequential

    for child in children:
        is_chainable = isinstance(child, _PRUNABLE) and (
            allow_conv or isinstance(child, nn.Linear)
        )
        if is_chainable:
            prunable_at_level.append(child)
            intermediates_at_level.append(current_intermediates)
            current_intermediates = []
        elif isinstance(child, _PASSTHROUGH_TYPES):
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
                current_intermediates.append(child)
        else:
            # Non-passthrough: flush chain and recurse
            _flush_chain(prunable_at_level, intermediates_at_level, chains)
            prunable_at_level = []
            intermediates_at_level = []
            current_intermediates = []
            _find_chains_recursive(child, chains)

    _flush_chain(prunable_at_level, intermediates_at_level, chains)


def _flush_chain(
    prunable: list[PrunableLayer],
    intermediates: list[list[nn.Module]],
    chains: list[tuple[PrunableLayer, PrunableLayer, list[nn.Module]]],
) -> None:
    """Convert a sequence of prunable layers into pairs."""
    if len(prunable) >= 2:
        for i in range(len(prunable) - 1):
            # intermediates[i+1] are the BN layers between prunable[i] and prunable[i+1]
            chains.append((prunable[i], prunable[i + 1], intermediates[i + 1]))
