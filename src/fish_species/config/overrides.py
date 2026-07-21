from __future__ import annotations

from copy import deepcopy
from typing import Any
import yaml

def parse_scalar(value: str) -> Any:
    """Parse the scalar syntax accepted by the legacy dotted-key CLI."""
    value = value.strip()

    try:
        return yaml.safe_load(value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid override value: {value!r}") from exc


def set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a dot-separated configuration key in place."""
    parts = key.split(".")
    current = config

    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(
                f"Cannot set {key!r}: {part!r} already contains a non-mapping value"
            )
        current = child

    current[parts[-1]] = value


def apply_overrides(
    config: dict[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    """Return a deep-copied config with legacy ``key=value`` overrides."""
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        set_nested(result, key, parse_scalar(value))
    return result
