"""Stable identifiers and output-directory names for training runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def short_hash(obj: Any, length: int = 8) -> str:
    """Return the historical deterministic identifier for a JSON value."""
    if length < 1:
        raise ValueError("Hash length must be positive")
    text = json.dumps(obj, sort_keys=True)
    # MD5 is used only as a stable, compact identifier; it is not a security hash.
    return hashlib.md5(text.encode()).hexdigest()[:length]


def make_run_name(cfg: dict[str, Any]) -> str:
    """Build the byte-compatible historical run-directory name."""
    parts = [
         'lr',
        cfg['training']['lr'],
        'hloss',
        cfg['multi_task']['hierarchy_loss']['enabled'],
        short_hash(cfg),
    ]
    return "_".join(str(part) for part in parts)


__all__ = ["make_run_name", "short_hash"]
