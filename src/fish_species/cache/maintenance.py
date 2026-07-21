"""Exact, testable persistent-cache build and verification contracts."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import os
import shutil
import socket
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..config.loading import load_config


READY_MARKER = "CACHE_READY"
LOCK_FILE = "CACHE_BUILD.lock"
MANIFEST_FILE = "cache_manifest.txt"


class CacheMaintenanceError(RuntimeError):
    """The cache could not be built or verified without ambiguity."""


@dataclass(frozen=True)
class CacheBuildResult:
    status: str
    cache_dir: str
    ready_marker: str
    manifest_path: str
    rows: int | None = None
    cached_rows: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolved_inputs(
    config_path: str | Path,
    data_root: str | Path,
    metadata_csv: str | Path,
    cache_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    config = Path(config_path).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    metadata = Path(metadata_csv).expanduser().resolve()
    cache = Path(cache_dir).expanduser().absolute()
    if not config.is_file():
        raise ValueError(f"config does not exist: {config}")
    if not data.is_dir():
        raise ValueError(f"data root is not a directory: {data}")
    if not metadata.is_file():
        raise ValueError(f"metadata CSV does not exist: {metadata}")
    if cache == Path(cache.anchor) or cache == data:
        raise ValueError("cache directory must be a dedicated child directory")
    if cache.exists() and cache.is_symlink():
        raise ValueError("cache directory must not be a symlink")
    return config, data, metadata, cache.resolve(strict=False)


def _runtime_config(
    source: dict[str, Any],
    *,
    data_root: Path,
    metadata_csv: Path,
    image_col: str,
    cache_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(source)
    config.setdefault("data", {}).update(
        root_dir=str(data_root),
        metadata_csv=str(metadata_csv),
        image_col=image_col,
    )
    config.setdefault("cache", {}).update(
        enabled=True,
        cache_dir=str(cache_dir),
        dir=str(cache_dir),
        root_dir=str(cache_dir),
        root_dir_cache=str(cache_dir),
    )
    return config


def _remove_stale_payload(cache_dir: Path, lock_path: Path) -> None:
    for child in cache_dir.iterdir():
        if child == lock_path:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise CacheMaintenanceError(f"unsupported cache entry: {child}")


def _atomic_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _manifest_text(
    *,
    created_utc: str,
    host: str,
    config_path: Path,
    data_root: Path,
    metadata_csv: Path,
    image_col: str,
    rows: int,
    cached_rows: int,
) -> str:
    return "\n".join(
        [
            f"created_utc={created_utc}",
            f"host={host}",
            f"config={config_path}",
            f"config_sha256={hashlib.sha256(config_path.read_bytes()).hexdigest()}",
            f"data_root={data_root}",
            f"metadata_csv={metadata_csv}",
            f"image_col={image_col}",
            f"rows={rows}",
            f"cached_rows={cached_rows}",
        ]
    ) + "\n"


def build_persistent_cache(
    config_path: str | Path,
    *,
    data_root: str | Path,
    metadata_csv: str | Path,
    cache_dir: str | Path,
    image_col: str = "rel_path_seg",
    force: bool = False,
    prepare: Callable[[dict[str, Any]], pd.DataFrame] | None = None,
    builder: Callable[[dict[str, Any], pd.DataFrame], pd.DataFrame] | None = None,
    now: Callable[[], datetime] | None = None,
    hostname: Callable[[], str] | None = None,
) -> CacheBuildResult:
    """Build a complete persistent cache, marking readiness only after success."""

    config_path, data_root, metadata_csv, cache_dir = _resolved_inputs(
        config_path, data_root, metadata_csv, cache_dir
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / LOCK_FILE
    ready_path = cache_dir / READY_MARKER
    manifest_path = cache_dir / MANIFEST_FILE

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if force:
            _remove_stale_payload(cache_dir, lock_path)
        elif ready_path.is_file():
            return CacheBuildResult(
                status="already_ready",
                cache_dir=str(cache_dir),
                ready_marker=str(ready_path),
                manifest_path=str(manifest_path),
            )

        if prepare is None:
            from ..data.metadata import prepare_metadata

            prepare = prepare_metadata
        if builder is None:
            from src.cache import build_image_cache

            builder = build_image_cache

        source_config = load_config(config_path)
        runtime_config = _runtime_config(
            source_config,
            data_root=data_root,
            metadata_csv=metadata_csv,
            image_col=image_col,
            cache_dir=cache_dir,
        )
        try:
            metadata = prepare(runtime_config)
        except Exception as exc:
            raise CacheMaintenanceError(f"metadata preparation failed: {exc}") from exc
        if metadata is None or len(metadata) == 0:
            raise CacheMaintenanceError("prepared metadata contains no rows")
        try:
            cached = builder(runtime_config, metadata)
        except Exception as exc:
            raise CacheMaintenanceError(f"cache builder failed: {exc}") from exc
        if cached is None:
            raise CacheMaintenanceError("cache builder returned None")
        if "_cached_image_path" not in cached.columns:
            raise CacheMaintenanceError(
                "cache builder did not produce '_cached_image_path'"
            )
        rows = len(cached)
        cached_rows = int(cached["_cached_image_path"].notna().sum())
        if cached_rows != rows:
            raise CacheMaintenanceError(
                f"cache is incomplete: {rows - cached_rows} of {rows} rows missing"
            )

        timestamp = (now or (lambda: datetime.now(timezone.utc)))()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        manifest = _manifest_text(
            created_utc=timestamp.isoformat(),
            host=(hostname or socket.gethostname)(),
            config_path=config_path,
            data_root=data_root,
            metadata_csv=metadata_csv,
            image_col=image_col,
            rows=rows,
            cached_rows=cached_rows,
        )
        _atomic_text(manifest_path, manifest)
        _atomic_text(ready_path, "")
        return CacheBuildResult(
            status="built",
            cache_dir=str(cache_dir),
            ready_marker=str(ready_path),
            manifest_path=str(manifest_path),
            rows=rows,
            cached_rows=cached_rows,
        )


def verify_persistent_cache(cache_dir: str | Path) -> CacheBuildResult:
    """Verify marker and manifest consistency without opening cached images."""

    cache = Path(cache_dir).expanduser().resolve()
    ready = cache / READY_MARKER
    manifest = cache / MANIFEST_FILE
    if not cache.is_dir() or not ready.is_file() or not manifest.is_file():
        raise CacheMaintenanceError(f"persistent cache is not ready: {cache}")
    fields: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    try:
        rows = int(fields["rows"])
        cached_rows = int(fields["cached_rows"])
    except (KeyError, ValueError) as exc:
        raise CacheMaintenanceError("cache manifest has invalid row counts") from exc
    if rows < 1 or cached_rows != rows:
        raise CacheMaintenanceError("cache manifest records an incomplete cache")
    return CacheBuildResult(
        status="ready",
        cache_dir=str(cache),
        ready_marker=str(ready),
        manifest_path=str(manifest),
        rows=rows,
        cached_rows=cached_rows,
    )
