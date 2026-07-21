from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def annotate_long_tail_groups(
    df: pd.DataFrame,
    species_col: str,
    head_min_samples: int = 11,
) -> pd.DataFrame:
    """Attach immutable frequency cohorts derived from the full labeled set."""
    if species_col not in df.columns or df[species_col].isna().any():
        raise ValueError(f"Long-tail grouping requires complete {species_col!r} labels")
    result = df.copy()
    counts = result[species_col].value_counts()
    result["__species_count__"] = result[species_col].map(counts).astype(int)
    result["__long_tail_group__"] = np.where(
        result["__species_count__"] >= head_min_samples, "head", "tail"
    )
    counts_for_rows = result["__species_count__"]
    result["__shot_group__"] = np.select(
        [
            counts_for_rows.between(2, 5),
            counts_for_rows.between(6, 20),
            counts_for_rows > 20,
        ],
        ["few_2_to_5", "medium_6_to_20", "many_over_20"],
        default="outside_reported_shot_groups",
    )
    return result


def make_long_tail_splits(
    df: pd.DataFrame,
    species_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    head_min_samples: int = 11,
    root_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split every species while reserving scarce tail examples carefully.

    Head species use the requested fractions. Tail species with at least three
    images contribute to both validation and test; two-image species retain one
    training image and contribute the other to test.
    """
    if head_min_samples < 2:
        raise ValueError("head_min_samples must be at least 2")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("Long-tail test/validation fractions must be in (0,1) and sum below 1")
    annotated = annotate_long_tail_groups(df, species_col, head_min_samples)
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    holdout_fraction = test_size + val_size

    for _, class_frame in annotated.groupby(species_col, sort=True):
        indices = class_frame.index.to_numpy(copy=True)
        rng.shuffle(indices)
        count = len(indices)
        is_head = count >= head_min_samples
        if count == 1:
            train_indices.extend(indices.tolist())
            continue
        if not is_head and count == 2:
            train_indices.append(int(indices[0]))
            test_indices.append(int(indices[1]))
            continue

        minimum_holdout = 1 if is_head else 2
        requested = max(minimum_holdout, int(round(count * holdout_fraction)))
        n_holdout = min(count - 1, requested)
        n_test = int(round(n_holdout * test_size / holdout_fraction))
        n_test = min(max(n_test, 1), n_holdout)
        n_val = n_holdout - n_test
        if n_val == 0 and n_holdout >= 2:
            n_val, n_test = 1, n_holdout - 1
        test_indices.extend(indices[:n_test].tolist())
        val_indices.extend(indices[n_test:n_test + n_val].tolist())
        train_indices.extend(indices[n_holdout:].tolist())

    for values in (train_indices, val_indices, test_indices):
        rng.shuffle(values)
    train_df = annotated.loc[train_indices].reset_index(drop=True)
    val_df = annotated.loc[val_indices].reset_index(drop=True)
    test_df = annotated.loc[test_indices].reset_index(drop=True)
    if set(train_df[species_col].unique()) != set(annotated[species_col].unique()):
        raise AssertionError("Every species must retain at least one training image")

    if root_dir is not None:
        split_dir = Path(root_dir) / "split_csv"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train_split.csv", index=False)
        val_df.to_csv(split_dir / "val_split.csv", index=False)
        test_df.to_csv(split_dir / "test_split.csv", index=False)
    return train_df, val_df, test_df


def make_individual_level_splits(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    root_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by individual/group, not by image row.

    This prevents images of the same fish from appearing in both train and test sets.
    Assumes each individual belongs to one target class.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    missing_columns = {target_col, group_col}.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Missing split columns: {sorted(missing_columns)}")

    group_df = (
        df[[group_col, target_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    counts_per_group = group_df.groupby(group_col)[target_col].nunique()
    bad_groups = counts_per_group[counts_per_group > 1]

    if len(bad_groups) > 0:
        raise ValueError(
            "Some groups have more than one target label. "
            "Check barcode/group parsing."
        )

    # One row per individual
    group_df = group_df.drop_duplicates(group_col).reset_index(drop=True)

    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    trainval_idx, test_idx = next(
        splitter_test.split(group_df[group_col], group_df[target_col])
    )

    group_trainval = group_df.iloc[trainval_idx].reset_index(drop=True)
    group_test = group_df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1.0 - test_size)

    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=seed + 1,
    )

    train_idx, val_idx = next(
        splitter_val.split(group_trainval[group_col], group_trainval[target_col])
    )

    group_train = group_trainval.iloc[train_idx]
    group_val = group_trainval.iloc[val_idx]

    train_groups = set(group_train[group_col])
    val_groups = set(group_val[group_col])
    test_groups = set(group_test[group_col])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    if root_dir is not None:
        split_dir = Path(root_dir) / "split_csv"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train_split.csv", index=False)
        val_df.to_csv(split_dir / "val_split.csv", index=False)
        test_df.to_csv(split_dir / "test_split.csv", index=False)
        print(f"Saved train/val/test splits to {split_dir}")

    return train_df, val_df, test_df


def make_class_stratified_splits(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    root_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split image rows while retaining at least one training row per class.

    Fish images do not carry meaningful individual identifiers.  This splitter
    therefore treats images as the sampling unit and labels as the unit that
    must be represented.  Rare classes are handled deterministically instead
    of being rejected by ``StratifiedShuffleSplit``: every class keeps at least
    one training image, and any remaining rows are divided between validation
    and test in the requested proportions.
    """
    if not 0 <= test_size < 1:
        raise ValueError("test_size must be between 0 (inclusive) and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 (inclusive) and 1")
    if test_size + val_size <= 0:
        raise ValueError("At least one of test_size and val_size must be positive")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")
    if target_col not in df.columns:
        raise KeyError(f"Missing split target column: {target_col}")
    if df[target_col].isna().any():
        raise ValueError(
            f"Class-stratified splitting requires a non-missing {target_col!r}. "
            "Use a fallback split label such as species-where-known, otherwise genus."
        )

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    holdout_fraction = test_size + val_size

    # Sorting makes the result stable even if pandas changes group iteration
    # details.  Randomness is restricted to ordering rows inside each class.
    for _, class_frame in df.groupby(target_col, sort=True):
        indices = class_frame.index.to_numpy(copy=True)
        rng.shuffle(indices)
        available = max(len(indices) - 1, 0)
        requested_holdout = max(1, int(round(len(indices) * holdout_fraction)))
        n_holdout = min(available, requested_holdout)

        if n_holdout == 0:
            train_indices.extend(indices.tolist())
            continue

        n_test = int(round(n_holdout * test_size / holdout_fraction))
        n_test = min(max(n_test, 0), n_holdout)
        n_val = n_holdout - n_test
        if val_size > 0 and n_val == 0 and n_holdout > 1:
            n_val, n_test = 1, n_holdout - 1
        if test_size > 0 and n_test == 0 and n_holdout > 1:
            n_test, n_val = 1, n_holdout - 1

        test_indices.extend(indices[:n_test].tolist())
        val_indices.extend(indices[n_test:n_test + n_val].tolist())
        train_indices.extend(indices[n_holdout:].tolist())

    # Shuffle split order so loaders do not inherit class-block ordering.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    test_df = df.loc[test_indices].reset_index(drop=True)

    missing_from_train = set(df[target_col].unique()) - set(
        train_df[target_col].unique()
    )
    if missing_from_train:
        raise AssertionError(
            f"Training split lost {len(missing_from_train)} classes unexpectedly."
        )

    if root_dir is not None:
        split_dir = Path(root_dir) / "split_csv"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train_split.csv", index=False)
        val_df.to_csv(split_dir / "val_split.csv", index=False)
        test_df.to_csv(split_dir / "test_split.csv", index=False)
        print(f"Saved train/val/test splits to {split_dir}")

    return train_df, val_df, test_df
