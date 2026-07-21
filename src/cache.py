from __future__ import annotations

import hashlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.fish_species.data.cropping import (
    foreground_bbox_from_image,
    foreground_bbox_from_mask,
    pad_square_bbox,
)
from src.fish_species.data.image_validation import resolve_path


def _file_stamp(path: Path) -> str:
    if not path.exists():
        return "missing"

    stat = path.stat()
    return f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"


def _make_cache_key(
    image_path: Path,
    cfg: dict[str, Any],
) -> str:
    data_cfg = cfg["data"]

    text = "|".join([
        _file_stamp(image_path),
        str(data_cfg.get("image_col")),
        str(cfg.get("preprocessing", {}).get("image_size", 224)),
        str(data_cfg.get("crop_to_foreground", False)),
        str(data_cfg.get("crop_pad", 0.0)),
        str(cfg.get("cache", {}).get("format", "png")),
    ])
    # Preserve existing cache keys; this digest is an identifier, not a security hash.
    return hashlib.md5(text.encode()).hexdigest()


def _cache_one_image(
    args: tuple[int, dict[str, Any], dict[str, Any]],
) -> tuple[int, str | None, str, str | None]:
    idx, row_dict, cfg = args

    data_cfg = cfg["data"]
    cache_cfg = cfg.get("cache", {})

    root_dir = Path(data_cfg["root_dir"])

    image_col = data_cfg["image_col"]
    mask_col = data_cfg.get("mask_col")

    image_size = int(cfg.get("preprocessing", {}).get("image_size", 224))
    crop_to_foreground = bool(data_cfg.get("crop_to_foreground", True))
    crop_pad = float(data_cfg.get("crop_pad", 0.15))

    fmt = cache_cfg.get("format", "png").lower()
    rebuild = bool(cache_cfg.get("rebuild", False))

    cache_dir = Path(cache_cfg.get("dir", "cache/images"))
    root_dir_cache = Path(cache_cfg.get("root_dir_cache", root_dir))
    if not cache_dir.is_absolute():
        cache_dir = root_dir_cache / cache_dir

    image_path = resolve_path(root_dir, row_dict[image_col])

    if not image_path.exists():
        return idx, None, "missing_image", str(image_path)

    mask_path = None
    if mask_col is not None and mask_col in row_dict and pd.notna(row_dict[mask_col]):
        candidate = resolve_path(root_dir, row_dict[mask_col])
        if candidate.exists():
            mask_path = candidate

    key = _make_cache_key(image_path, cfg)

    suffix = ".jpg" if fmt in {"jpg", "jpeg"} else ".png"
    out_path = cache_dir / key[:2] / f"{key}{suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not rebuild:
        return idx, str(out_path), "exists", None

    try:
        with Image.open(image_path) as source:
            img = source.convert("RGB")

        if crop_to_foreground:
            bbox = None

            if mask_path is not None:
                bbox = foreground_bbox_from_mask(mask_path)

            if bbox is None:
                bbox = foreground_bbox_from_image(img)

            if bbox is not None:
                w, h = img.size
                bbox = pad_square_bbox(bbox, w, h, crop_pad)
                img = img.crop(bbox)

        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=out_path.parent,
            prefix=f".{out_path.stem}.",
            suffix=out_path.suffix,
        )
        os.close(file_descriptor)
        tmp_path = Path(temporary_name)
        try:
            if suffix == ".jpg":
                img.save(tmp_path, quality=100)
            else:
                img.save(tmp_path)
            os.replace(tmp_path, out_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        return idx, str(out_path), "created", None

    except Exception as exc:
        return idx, None, "error", repr(exc)


def build_image_cache(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Build deterministic image cache for faster training.

    Adds:
    - _cached_image_path
    - _cache_status

    The cache stores cropped/resized images, but random training augmentations
    are still applied later by the Dataset transforms.
    """

    cache_cfg = cfg.get("cache", {})

    if not cache_cfg.get("enabled", False):
        return df

    df = df.reset_index(drop=True).copy()

    num_workers = int(
        cache_cfg.get(
            "num_workers",
            cfg.get("training", {}).get("num_workers", 4),
        )
    )
    if num_workers < 1:
        raise ValueError("cache.num_workers must be at least 1")

    tasks = [
        (idx, row.to_dict(), cfg)
        for idx, row in df.iterrows()
    ]

    cached_paths = [None] * len(df)
    statuses = [None] * len(df)
    errors = [None] * len(df)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_cache_one_image, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building image cache"):
            idx, cache_path, status, error = fut.result()
            cached_paths[idx] = cache_path
            statuses[idx] = status
            errors[idx] = error

    df["_cached_image_path"] = cached_paths
    df["_cache_status"] = statuses
    df["_cache_error"] = errors

    n_ok = df["_cached_image_path"].notna().sum()
    n_bad = len(df) - n_ok

    print(f"Cached/available images: {n_ok}/{len(df)}")
    print(f"Failed images: {n_bad}")
    print(df["_cache_status"].value_counts(dropna=False))

    if n_bad > 0:
        print("\nExamples of failed cache rows:")
        display_cols = [
            cfg["data"].get("group_col"),
            cfg["data"].get("target_col"),
            cfg["data"].get("image_col"),
            "_cache_status",
            "_cache_error",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        print(df[df["_cached_image_path"].isna()][display_cols].head(10))

    return df
