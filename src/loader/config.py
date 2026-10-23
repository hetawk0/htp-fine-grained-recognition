"""YAML config loading with nested merge (meddef-style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


def deep_update(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)  # type: ignore[arg-type]
        else:
            base[k] = v
    return dict(base)


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    # Optional inheritance: base: other.yaml
    base_name = cfg.pop("base", None)
    if base_name:
        base_path = path.parent / base_name
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)

    if overrides:
        cfg = deep_update(cfg, dict(overrides))
    cfg["_config_path"] = str(path.resolve())
    return cfg
