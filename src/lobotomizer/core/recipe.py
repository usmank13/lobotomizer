"""Recipe loading from YAML."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml

from lobotomizer.core.pipeline import Pipeline
from lobotomizer.stages.base import Stage

# Registry of stage type names to classes
_STAGE_REGISTRY: dict[str, type[Stage]] = {}

_RECIPES_DIR = pathlib.Path(__file__).parent.parent / "recipes"


def _ensure_registry() -> None:
    """Lazily populate the stage registry."""
    if _STAGE_REGISTRY:
        return
    from lobotomizer.stages.prune import Prune
    from lobotomizer.stages.quantize import Quantize

    _STAGE_REGISTRY["prune"] = Prune
    _STAGE_REGISTRY["quantize"] = Quantize


def load_recipe(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a compression recipe from a YAML file."""
    p = pathlib.Path(path)
    with p.open() as f:
        return yaml.safe_load(f) or {}


def build_pipeline_from_recipe(recipe: str | pathlib.Path | dict[str, Any]) -> Pipeline:
    """Build a Pipeline from a recipe name, path, or dict.

    If *recipe* is a string without path separators and no file extension,
    it is looked up in the built-in recipes directory.
    """
    _ensure_registry()

    if isinstance(recipe, dict):
        data = recipe
    else:
        recipe_path = pathlib.Path(recipe)
        if not recipe_path.suffix and not recipe_path.parent.name:
            # Treat as built-in recipe name
            recipe_path = _RECIPES_DIR / f"{recipe}.yaml"
        data = load_recipe(recipe_path)

    stages: list[Stage] = []
    for stage_cfg in data.get("stages", []):
        cfg = dict(stage_cfg)
        type_name = cfg.pop("type")
        cls = _STAGE_REGISTRY.get(type_name)
        if cls is None:
            raise ValueError(
                f"Unknown stage type '{type_name}'. Registered: {list(_STAGE_REGISTRY)}"
            )
        stages.append(cls(**cfg))

    return Pipeline(stages)
