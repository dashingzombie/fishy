"""Stable lightweight writers for scientific result metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Write the historical two-space-indented JSON contract."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
        handle.write("\n")


__all__ = ["save_json"]
