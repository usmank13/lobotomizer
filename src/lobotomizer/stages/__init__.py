"""Compression stages."""
from lobotomizer.stages.base import PipelineContext, Stage, StageResult
from lobotomizer.stages.prune import Prune
from lobotomizer.stages.quantize import Quantize
from lobotomizer.stages.structured_prune import StructuredPrune

__all__ = ["Stage", "PipelineContext", "StageResult", "Prune", "Quantize", "StructuredPrune"]
