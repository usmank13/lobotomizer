"""Low-rank decomposition via truncated SVD for Linear and Conv2d layers."""
from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn

from lobotomizer.stages.base import PipelineContext, Stage

logger = logging.getLogger(__name__)


class LowRank(Stage):
    """Replace weight matrices with low-rank approximations using truncated SVD.

    For a Linear layer with weight W of shape (out, in), this decomposes it
    into two smaller matrices: W ≈ U @ V where U is (out, rank) and V is
    (rank, in). The single Linear layer is replaced with two sequential
    Linear layers, reducing parameter count when rank < (out * in) / (out + in).

    For Conv2d layers with 1×1 kernels, the same decomposition applies to
    the (out_channels, in_channels) weight matrix. For larger kernels,
    a spatial decomposition is used: the (out, in, kH, kW) weight is reshaped
    to (out, in*kH*kW), decomposed, then the V matrix is reshaped back to
    a Conv2d with kernel (kH, kW) and the U matrix becomes a 1×1 Conv2d.

    Parameters
    ----------
    rank_fraction : float
        Fraction of the maximum possible rank to keep (0.0 to 1.0).
        E.g., 0.5 keeps half the singular values.
    min_rank : int
        Minimum rank to preserve (avoids degenerate decompositions).
    criterion : "energy" or "fraction"
        - "fraction": keep ``rank_fraction`` of the full rank.
        - "energy": keep enough singular values to preserve ``rank_fraction``
          of the total spectral energy (sum of squared singular values).
    exclude_layers : set[str] | list[str] | None
        Named layers to exclude from decomposition.
    min_compression : float
        Only decompose a layer if the resulting parameter count is at most
        this fraction of the original (default 0.9 = must save at least 10%).
    """

    def __init__(
        self,
        rank_fraction: float = 0.5,
        min_rank: int = 1,
        criterion: Literal["fraction", "energy"] = "fraction",
        exclude_layers: set[str] | list[str] | None = None,
        min_compression: float = 0.9,
    ) -> None:
        if not 0.0 < rank_fraction <= 1.0:
            raise ValueError(f"rank_fraction must be in (0, 1], got {rank_fraction}")
        if min_rank < 1:
            raise ValueError(f"min_rank must be >= 1, got {min_rank}")
        if criterion not in ("fraction", "energy"):
            raise ValueError(f"criterion must be 'fraction' or 'energy', got '{criterion}'")
        self._rank_fraction = rank_fraction
        self._min_rank = min_rank
        self._criterion = criterion
        self._exclude_layers: set[str] = set(exclude_layers) if exclude_layers else set()
        self._min_compression = min_compression

    @property
    def name(self) -> str:
        return f"low_rank({self._criterion}, {self._rank_fraction})"

    def validate(self, model: nn.Module) -> list[str]:
        warnings: list[str] = []
        decomposable = any(
            isinstance(m, (nn.Linear, nn.Conv2d)) for m in model.modules()
        )
        if not decomposable:
            warnings.append("Model has no Linear or Conv2d layers for low-rank decomposition.")
        return warnings

    def estimate_impact(self, model: nn.Module) -> dict:
        total_params = sum(p.numel() for p in model.parameters())
        decomposable = sum(
            p.numel()
            for m in model.modules()
            if isinstance(m, (nn.Linear, nn.Conv2d))
            for p in m.parameters()
        )
        return {
            "total_params": total_params,
            "decomposable_params": decomposable,
            "estimated_fraction_kept": self._rank_fraction,
        }

    def apply(self, model: nn.Module, context: PipelineContext) -> nn.Module:
        replacements: list[tuple[str, nn.Module, nn.Module]] = []

        for name, module in model.named_modules():
            if name in self._exclude_layers:
                logger.info("Excluding layer '%s' from decomposition (user-specified).", name)
                continue

            if isinstance(module, nn.Linear):
                replacement = self._decompose_linear(module)
                if replacement is not None:
                    replacements.append((name, module, replacement))
                    logger.info(
                        "Decomposed Linear '%s': (%d, %d) → rank %d",
                        name, module.out_features, module.in_features,
                        _get_rank_from_sequential(replacement),
                    )

            elif isinstance(module, nn.Conv2d):
                replacement = self._decompose_conv2d(module)
                if replacement is not None:
                    replacements.append((name, module, replacement))
                    logger.info("Decomposed Conv2d '%s'", name)

        # Apply replacements
        for name, old_module, new_module in replacements:
            _replace_module(model, name, new_module)

        return model

    # ------------------------------------------------------------------
    # Decomposition methods
    # ------------------------------------------------------------------

    def _compute_rank(self, weight_2d: torch.Tensor) -> int:
        """Compute target rank for a 2D weight matrix."""
        max_rank = min(weight_2d.shape)

        if self._criterion == "fraction":
            rank = max(self._min_rank, int(max_rank * self._rank_fraction))
        else:  # energy
            U, S, Vh = torch.linalg.svd(weight_2d, full_matrices=False)
            total_energy = (S ** 2).sum().item()
            cumulative = torch.cumsum(S ** 2, dim=0)
            threshold = total_energy * self._rank_fraction
            rank = int((cumulative >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
            rank = max(rank, self._min_rank)

        return min(rank, max_rank)

    def _decompose_linear(self, layer: nn.Linear) -> nn.Sequential | None:
        """Decompose a Linear layer into two smaller Linear layers."""
        W = layer.weight.data  # (out, in)
        out_features, in_features = W.shape

        rank = self._compute_rank(W)

        # Check if decomposition actually saves parameters
        original_params = out_features * in_features + (out_features if layer.bias is not None else 0)
        new_params = rank * in_features + out_features * rank + (out_features if layer.bias is not None else 0)
        if new_params >= original_params * self._min_compression:
            return None

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # W ≈ U[:, :rank] @ diag(S[:rank]) @ Vh[:rank, :]
        # Split as: first_layer = sqrt(S) * Vh, second_layer = U * sqrt(S)
        sqrt_S = torch.sqrt(S[:rank])
        V_layer = nn.Linear(in_features, rank, bias=False)
        V_layer.weight = nn.Parameter((sqrt_S.unsqueeze(1) * Vh[:rank, :]))

        U_layer = nn.Linear(rank, out_features, bias=layer.bias is not None)
        U_layer.weight = nn.Parameter(U[:, :rank] * sqrt_S.unsqueeze(0))
        if layer.bias is not None:
            U_layer.bias = nn.Parameter(layer.bias.data.clone())

        return nn.Sequential(V_layer, U_layer)

    def _decompose_conv2d(self, layer: nn.Conv2d) -> nn.Sequential | None:
        """Decompose a Conv2d layer into two Conv2d layers via SVD.

        For kernel (kH, kW), reshape weight to (out_channels, in_channels*kH*kW),
        decompose, then reconstruct as:
          1. Conv2d(in_channels, rank, kernel_size=(kH, kW), ...) — no bias
          2. Conv2d(rank, out_channels, kernel_size=1) — carries original bias
        """
        W = layer.weight.data  # (out_ch, in_ch, kH, kW)
        out_ch, in_ch, kH, kW = W.shape

        W_2d = W.reshape(out_ch, -1)  # (out_ch, in_ch*kH*kW)
        rank = self._compute_rank(W_2d)

        # Check compression ratio
        original_params = W.numel() + (layer.bias.numel() if layer.bias is not None else 0)
        new_params = (rank * in_ch * kH * kW) + (out_ch * rank) + (out_ch if layer.bias is not None else 0)
        if new_params >= original_params * self._min_compression:
            return None

        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)
        sqrt_S = torch.sqrt(S[:rank])

        # First conv: (in_ch, kH, kW) → rank channels
        V_weight = (sqrt_S.unsqueeze(1) * Vh[:rank, :]).reshape(rank, in_ch, kH, kW)
        V_conv = nn.Conv2d(
            in_ch, rank, kernel_size=(kH, kW),
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups,
            bias=False,
        )
        V_conv.weight = nn.Parameter(V_weight)

        # Second conv: 1×1, rank → out_ch
        U_weight = (U[:, :rank] * sqrt_S.unsqueeze(0)).reshape(out_ch, rank, 1, 1)
        U_conv = nn.Conv2d(rank, out_ch, kernel_size=1, bias=layer.bias is not None)
        U_conv.weight = nn.Parameter(U_weight)
        if layer.bias is not None:
            U_conv.bias = nn.Parameter(layer.bias.data.clone())

        return nn.Sequential(V_conv, U_conv)


# ======================================================================
# Helpers
# ======================================================================

def _replace_module(model: nn.Module, target_name: str, new_module: nn.Module) -> None:
    """Replace a named module in the model hierarchy."""
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


def _get_rank_from_sequential(seq: nn.Sequential) -> int:
    """Extract the rank from a decomposed Sequential(V, U)."""
    first = seq[0]
    if isinstance(first, nn.Linear):
        return first.out_features
    elif isinstance(first, nn.Conv2d):
        return first.out_channels
    return -1
