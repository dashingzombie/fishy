from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .image_validation import is_valid_image, resolve_path
from .taxonomy import add_fish_taxonomy, derive_genus_from_species


def _data_path(data_cfg: dict, key: str, default: str) -> Path:
    path = Path(str(data_cfg.get(key, default)))
    if path.is_absolute():
        return path
    return Path(str(data_cfg.get("metadata_dir", "../data"))) / path


def _fish_image_value(filename: str, split_name: str, data_cfg: dict) -> str:
    """Build a path relative to ``data.root_dir`` for one fish image."""
    image_dirs = data_cfg.get("image_dirs", {}) or {}
    if not isinstance(image_dirs, dict):
        raise TypeError("data.image_dirs must be a mapping from split to directory")
    split_dir = str(image_dirs.get(split_name, "") or "")
    return str(Path(split_dir) / Path(str(filename)).name)


def _derive_fish_genus(species: pd.Series) -> pd.Series:
    return derive_genus_from_species(species)


def load_fish_training_metadata(cfg: dict) -> pd.DataFrame:
    """Load the labeled FishNet-style pickle/JSON training metadata."""
    data_cfg = cfg["data"]
    labels_path = _data_path(data_cfg, "labels_json", "label_train.json")
    split_dir = _data_path(data_cfg, "split_dir", "splits")
    split_files = data_cfg.get("split_files", {}) or {}
    train_file = split_files.get("train", "train.pkl")
    train_path = split_dir / str(train_file)

    with labels_path.open(encoding="utf-8") as handle:
        labels = json.load(handle)
    filenames = pd.read_pickle(train_path)
    if not isinstance(filenames, (list, tuple, pd.Series)):
        raise TypeError(f"Expected a filename sequence in {train_path}")

    missing = [name for name in filenames if str(name) not in labels]
    if missing:
        raise ValueError(
            f"{len(missing)} training filenames have no label in {labels_path}; "
            f"examples: {missing[:5]}"
        )

    frame = pd.DataFrame({"image_id": [str(name) for name in filenames]})
    frame["species_label"] = frame["image_id"].map(labels)
    frame["genus"] = _derive_fish_genus(frame["species_label"])
    frame["image_path"] = frame["image_id"].map(
        lambda name: _fish_image_value(name, "train", data_cfg)
    )
    frame["__taxon_for_split__"] = frame["species_label"]
    return frame


def load_fish_prediction_metadata(
    cfg: dict,
    split_names: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load unlabeled fish filename lists for image-only inference."""
    data_cfg = cfg["data"]
    split_dir = _data_path(data_cfg, "split_dir", "splits")
    split_files = data_cfg.get("split_files", {}) or {}
    requested = list(
        split_names
        if split_names is not None
        else (cfg.get("inference", {}) or {}).get("splits", ["test", "unseen"])
    )
    frames: dict[str, pd.DataFrame] = {}
    for split_name in requested:
        filename = split_files.get(split_name, f"{split_name}.pkl")
        path = split_dir / str(filename)
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured inference split {split_name!r} was not found: {path}"
            )
        names = pd.read_pickle(path)
        if not isinstance(names, (list, tuple, pd.Series)):
            raise TypeError(f"Expected a filename sequence in {path}")
        frame = pd.DataFrame({"image_id": [str(name) for name in names]})
        frame["image_path"] = frame["image_id"].map(
            lambda name, split=split_name: _fish_image_value(name, split, data_cfg)
        )
        frames[str(split_name)] = frame
    return frames


def _prepare_fish_metadata(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    df = load_fish_training_metadata(cfg)
    image_col = data_cfg["image_col"]
    df = df.dropna(subset=[image_col]).reset_index(drop=True)
    print(f"Initial labeled fish dataset size: {len(df)}")
    if data_cfg.get("validate_images", False):
        df = df[
            df[image_col].apply(
                lambda value: is_valid_image(resolve_path(data_cfg["root_dir"], value))
            )
        ].reset_index(drop=True)
        print(f"After removing invalid images, dataset size: {len(df)}")

    for task, column in data_cfg["target_cols"].items():
        if column not in df.columns or df[column].isna().any():
            raise ValueError(f"Labeled training data requires complete {task!r} labels")
    species_col = data_cfg["target_cols"].get("species")
    if species_col:
        df["__taxon_for_split__"] = df[species_col]
    return df


def _prepare_csv_metadata(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    data_cfg = cfg["data"]
    image_col = data_cfg["image_col"]
    group_col = data_cfg["group_col"]

    df = add_fish_taxonomy(df, cfg)
    df[group_col] = df[group_col].astype(str)
    if data_cfg.get("strip_final_number_from_group", False):
        df[group_col] = df[group_col].str.replace(r"_(\d+)$", "", regex=True)
    df = df.dropna(subset=[image_col, group_col]).reset_index(drop=True)
    print(f"Initial dataset size: {len(df)}")
    if data_cfg.get("validate_images", True):
        df = df[
            df[image_col].apply(
                lambda value: is_valid_image(resolve_path(data_cfg["root_dir"], value))
            )
        ].reset_index(drop=True)
        print(f"After removing invalid images, dataset size: {len(df)}")
    for task, column in data_cfg["target_cols"].items():
        if column not in df.columns or df[column].isna().any():
            raise ValueError(f"Labeled training data requires complete {task!r} labels")
    return df


def prepare_metadata(cfg: dict) -> pd.DataFrame:
    """Prepare either fish pickle/JSON data or the legacy CSV dataset."""
    dataset_format = str(
        cfg.get("data", {}).get("dataset_format", "csv")
    ).lower()
    if dataset_format == "fish_pickle":
        return _prepare_fish_metadata(cfg)
    if dataset_format == "csv":
        return _prepare_csv_metadata(cfg)
    raise ValueError(
        f"Unsupported data.dataset_format={dataset_format!r}; "
        "choose 'fish_pickle' or 'csv'."
    )
