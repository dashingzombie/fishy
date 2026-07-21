"""Size-limited readers for lightweight scientific result metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 10 * 1024 * 1024
MAX_TEXT_BYTES = 10 * 1024 * 1024
MAX_TABULAR_BYTES = 100 * 1024 * 1024
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def _read_limited(path: Path, max_bytes: int) -> str:
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        raise ValueError("checkpoint contents are not readable by result discovery")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"file is {size} bytes; read limit is {max_bytes} bytes")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(max_bytes + 1)


def load_json(path: str | Path, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    """Load a capped JSON object; checkpoint suffixes are always refused."""

    value = json.loads(_read_limited(Path(path), max_bytes))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_text(path: str | Path, max_bytes: int = MAX_TEXT_BYTES) -> str:
    return _read_limited(Path(path), max_bytes)


def load_csv_rows(
    path: str | Path,
    *,
    max_bytes: int = MAX_TABULAR_BYTES,
    max_rows: int | None = None,
) -> list[dict[str, str]]:
    """Read a capped CSV/TSV using only the standard library."""

    path = Path(path)
    text = _read_limited(path, max_bytes)
    dialect = "excel-tab" if path.suffix.lower() == ".tsv" else "excel"
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(reader):
        if max_rows is not None and row_index >= max_rows:
            break
        rows.append(dict(row))
    return rows
