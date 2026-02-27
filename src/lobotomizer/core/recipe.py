"""Recipe loading from YAML."""
from __future__ import annotations

import pathlib
from typing import Any

import yaml


def load_recipe(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a compression recipe from a YAML file."""
    p = pathlib.Path(path)
    with p.open() as f:
        return yaml.safe_load(f) or {}
