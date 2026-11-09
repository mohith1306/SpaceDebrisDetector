"""Loader for the experiment configuration (configs/physics_config.yaml).

The YAML file is the single source of truth for experiment parameters so that
the baseline and the physics-motivated run stay comparable. Scripts read it
here and allow explicit CLI flags to override individual values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "physics_config.yaml"


def load_physics_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the experiment YAML. Falls back to the repo-default location."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {cfg_path}. "
            "Expected configs/physics_config.yaml in the repository root."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {cfg_path}: expected a mapping")
    return data


def cfg_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value: cfg_get(cfg, 'training.epochs', 30)."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
