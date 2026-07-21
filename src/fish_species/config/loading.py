from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigLoadError(ValueError):
    """A configuration file or its inheritance chain cannot be resolved."""


def deep_merge(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input.

    Existing keys retain their base insertion position.  This matters for the
    established sweep/run-spec byte contract, whose override order follows YAML
    mapping order.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_extended(
    path: Path,
    stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    config_path = path.resolve()
    if config_path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, config_path))
        raise ConfigLoadError(f"Configuration extends cycle: {cycle}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigLoadError(f"Configuration not found: {config_path}") from exc

    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ConfigLoadError(
            f"Configuration must be a YAML mapping: {config_path}"
        )

    parent = config.pop("extends", None)
    if parent is None:
        return config
    if not isinstance(parent, str) or not parent.strip():
        raise ConfigLoadError(
            f"extends must be a non-empty path string: {config_path}"
        )

    parent_config = _load_extended(
        config_path.parent / parent,
        (*stack, config_path),
    )
    return deep_merge(parent_config, config)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, resolving an optional relative ``extends`` chain.

    Loading applies no defaults and does not mutate any source mapping.
    """
    return _load_extended(Path(path))


__all__ = ["ConfigLoadError", "deep_merge", "load_config"]
