"""Profile-aware loader assembly for the canonical trainer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os

import pandas as pd
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, DistributedSampler, Sampler, WeightedRandomSampler

from src.cache import build_image_cache
from src.splits import make_class_stratified_splits, make_individual_level_splits
from src.splits import make_long_tail_splits

from ..data.datasets import InferenceImageDataset, MultiTaskImageDataset
from ..data.labels import build_label_maps
from ..data.labels import get_target_cols
from ..data.labels import read_csvs_from_dir
from ..data.metadata import load_fish_prediction_metadata, prepare_metadata
from ..data.transforms import build_split_transform
from .modes import TrainingProfile
from .modes import get_profile


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    label_to_index_by_task: dict
    index_to_label_by_task: dict
    split_summary: dict
    train_df: object
    target_cols: dict
    test_loader_context: dict | None = None
    prediction_loaders: dict[str, DataLoader] | None = None
    prediction_frames: dict[str, object] | None = None
    head_val_loader: DataLoader | None = None
    head_test_loader: DataLoader | None = None
    tail_replay_loader: DataLoader | None = None
    species_counts: dict[str, int] | None = None
    stage_one_train_df: object | None = None
    tail_replay_df: object | None = None


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic inverse-frequency sampling partitioned across torchrun ranks."""

    def __init__(self, weights: torch.Tensor, seed: int) -> None:
        self.weights = weights.to(dtype=torch.double, device="cpu")
        self.seed = int(seed)
        self.epoch = 0
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.num_samples = int(math.ceil(len(weights) / self.world_size))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.num_samples * self.world_size,
            replacement=True,
            generator=generator,
        )
        return iter(indices[self.rank :: self.world_size].tolist())


def _distributed_training() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _build_cache_safely(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Build once on rank zero, then let other ranks open the completed cache."""
    if not (distributed.is_available() and distributed.is_initialized()):
        return build_image_cache(cfg, frame)
    cached = None
    if distributed.get_rank() == 0:
        cached = build_image_cache(cfg, frame)
    distributed.barrier()
    if distributed.get_rank() != 0:
        cached = build_image_cache(cfg, frame)
    distributed.barrier()
    assert cached is not None
    return cached


def _training_sampler(
    dataset,
    df: pd.DataFrame,
    cfg: dict,
    species_col: str,
):
    strategy = str((cfg.get("training", {}).get("sampling", {}) or {}).get(
        "strategy", "random"
    )).lower()
    distributed = _distributed_training()
    if strategy == "random":
        return DistributedSampler(
            dataset,
            shuffle=True,
            seed=int(cfg.get("seed", 0)),
            drop_last=False,
        ) if distributed else None
    if strategy != "weighted":
        raise ValueError("training.sampling.strategy must be 'random' or 'weighted'")
    counts = df[species_col].value_counts()
    weights = df[species_col].map(lambda label: 1.0 / float(counts[label]))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    tensor = torch.as_tensor(weights.to_numpy(), dtype=torch.double)
    if distributed:
        return DistributedWeightedSampler(tensor, int(cfg.get("seed", 0)))
    return WeightedRandomSampler(
        tensor, num_samples=len(df), replacement=True, generator=generator
    )


def _tail_replay_frame(train_df: pd.DataFrame, cfg: dict) -> pd.DataFrame | None:
    staged = (cfg.get("long_tail", {}) or {}).get("staged_training", {}) or {}
    if not bool(staged.get("enabled", False)):
        return None
    tail = train_df[train_df["__long_tail_group__"] == "tail"]
    head = train_df[train_df["__long_tail_group__"] == "head"]
    if tail.empty:
        raise ValueError("Staged long-tail training has no tail samples")
    replay_fraction = float(staged.get("head_replay_fraction", 0.25))
    if not 0 <= replay_fraction < 1:
        raise ValueError("long_tail.staged_training.head_replay_fraction must be in [0,1)")
    replay_count = int(round(len(tail) * replay_fraction / max(1.0 - replay_fraction, 1e-12)))
    replay = head.sample(
        n=replay_count,
        replace=replay_count > len(head),
        random_state=int(cfg.get("seed", 0)) + 73,
    ) if replay_count else head.iloc[0:0]
    return pd.concat([tail, replay], ignore_index=True).sample(
        frac=1.0, random_state=int(cfg.get("seed", 0)) + 74
    ).reset_index(drop=True)


def get_input_condition(cfg: dict) -> dict:
    raw = copy.deepcopy(cfg.get("input_condition", {}) or {})
    if not bool(raw.get("enabled", False)):
        return {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }

    transform_name = str(raw.get("transform", "original")).lower()
    condition = {
        "condition": str(
            raw.get("condition") or raw.get("name") or transform_name
        ),
        "feature": str(raw.get("feature", "baseline")),
        "transform": transform_name,
        "strength": float(raw.get("strength", 0.0)),
    }
    nested_parameters = raw.get("parameters", {}) or {}
    if not isinstance(nested_parameters, dict):
        raise TypeError("input_condition.parameters must be a mapping")
    parameter_keys = {
        "retention",
        "order",
        "diameter",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "grid_size",
        "seed",
    }
    for key in parameter_keys:
        value = raw.get(key, nested_parameters.get(key))
        if value is not None:
            condition[key] = value

    if transform_name == "saturation":
        condition["retention"] = float(condition.get("retention", 1.0))
        if not 0.0 <= condition["retention"] <= 1.0:
            raise ValueError(
                "input_condition.retention must be in [0, 1], got "
                f"{condition['retention']}."
            )
    elif transform_name == "channel_shuffle":
        order = condition.get("order", [2, 0, 1])
        condition["order"] = (
            [int(value.strip()) for value in order.split(",")]
            if isinstance(order, str)
            else [int(value) for value in order]
        )
    elif transform_name == "bilateral_filter":
        condition.update(
            diameter=int(condition["diameter"]),
            sigma_colour=float(condition["sigma_colour"]),
            sigma_space=float(condition["sigma_space"]),
        )
    elif transform_name == "gaussian_blur":
        condition["sigma"] = float(condition["sigma"])
    elif transform_name == "patch_shuffle":
        condition.update(
            grid_size=int(condition["grid_size"]),
            seed=int(condition.get("seed", cfg.get("seed", 0))),
        )
    elif transform_name not in {"original", "grayscale"}:
        raise ValueError(
            f"Unsupported input condition transform: {transform_name!r}."
        )

    return condition


def make_profile_loaders(cfg: dict, profile: TrainingProfile) -> LoaderBundle:
    df = prepare_metadata(cfg)
    target_cols = get_target_cols(cfg)
    group_col = cfg["data"]["group_col"]
    split_target_col = cfg["data"].get(
        "split_target_col", "__taxon_for_split__"
    )
    if split_target_col not in df.columns:
        raise ValueError(
            f"data.split_target_col={split_target_col!r} is not in the metadata "
            "dataframe. Use '__taxon_for_split__' or an existing column."
        )

    cache_enabled = cfg.get("cache", {}).get("enabled", False)
    if cache_enabled:
        df = _build_cache_safely(cfg, df)
        df = df[df["_cached_image_path"].notna()].reset_index(drop=True)
        image_col_for_dataset = "_cached_image_path"
        crop_to_foreground_for_dataset = False
    else:
        image_col_for_dataset = cfg["data"]["image_col"]
        crop_to_foreground_for_dataset = cfg["data"].get(
            "crop_to_foreground", True
        )

    if cfg["split"].get("use_predefined_splits", False):
        train_df, val_df, test_df = read_csvs_from_dir(
            cfg["split"]["predefined_split_dir"]
        )
    else:
        split_strategy = str(
            cfg["split"].get("strategy", "group_stratified")
        ).lower()
        split_kwargs = {
            "df": df,
            "target_col": split_target_col,
            "test_size": cfg["split"]["test_size"],
            "val_size": cfg["split"]["val_size"],
            "seed": cfg["seed"],
            "root_dir": (
                cfg["split"]["predefined_split_dir"]
                if cfg["split"].get("save_splits", False)
                else None
            ),
        }
        if split_strategy == "long_tail":
            species_col = target_cols.get("species")
            if species_col is None:
                raise ValueError("Long-tail splitting requires a species task")
            train_df, val_df, test_df = make_long_tail_splits(
                species_col=species_col,
                head_min_samples=int((cfg.get("long_tail", {}) or {}).get(
                    "head_min_samples", 11
                )),
                **{key: value for key, value in split_kwargs.items() if key != "target_col"},
            )
        elif split_strategy == "class_stratified":
            train_df, val_df, test_df = make_class_stratified_splits(
                **split_kwargs
            )
        elif split_strategy == "group_stratified":
            train_df, val_df, test_df = make_individual_level_splits(
                group_col=group_col,
                **split_kwargs,
            )
        else:
            raise ValueError(
                "split.strategy must be 'long_tail', 'class_stratified', or "
                "'group_stratified'"
            )

    if cfg["split"].get("use_predefined_splits", False) and cfg.get(
        "cache", {}
    ).get("enabled", False):
        print(f"Using predefined splits from {cfg['split']['predefined_split_dir']}")
        train_df = _build_cache_safely(cfg, train_df)
        val_df = _build_cache_safely(cfg, val_df)
        test_df = _build_cache_safely(cfg, test_df)
        train_df = train_df[
            train_df["_cached_image_path"].notna()
        ].reset_index(drop=True)
        val_df = val_df[val_df["_cached_image_path"].notna()].reset_index(
            drop=True
        )
        test_df = test_df[test_df["_cached_image_path"].notna()].reset_index(
            drop=True
        )

    label_to_index_by_task, index_to_label_by_task = build_label_maps(
        train_df, target_cols
    )
    preprocessing = copy.deepcopy(cfg.get("preprocessing", {}) or {})
    if not isinstance(preprocessing, dict):
        raise TypeError("preprocessing must be a mapping")
    if "image_size" not in preprocessing:
        preprocessing["image_size"] = cfg["data"]["image_size"]
    augmentation = copy.deepcopy(cfg.get("augmentation", {}) or {})
    if not isinstance(augmentation, dict):
        raise TypeError("augmentation must be a mapping")
    image_size = preprocessing["image_size"]
    input_condition = (
        get_input_condition(cfg)
        if profile.loader_mode == "condition"
        else {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }
    )

    train_tf = build_split_transform(
        split="train",
        preprocessing=preprocessing,
        augmentation=augmentation,
        condition=input_condition,
        original_colour_retention=1.0,
    )
    eval_tf = build_split_transform(
        split="validation",
        preprocessing=preprocessing,
        augmentation=augmentation,
        condition=input_condition,
        original_colour_retention=1.0,
    )

    common_kwargs = {
        "root_dir": cfg["data"]["root_dir"],
        "image_col": image_col_for_dataset,
        "target_cols": target_cols,
        "label_to_index_by_task": label_to_index_by_task,
        "mask_col": cfg["data"].get("mask_col"),
        "crop_to_foreground": crop_to_foreground_for_dataset,
        "crop_pad": cfg["data"].get("crop_pad", 0.15),
    }
    long_tail_enabled = "__long_tail_group__" in train_df.columns
    stage_one_df = (
        train_df[train_df["__long_tail_group__"] == "head"].reset_index(drop=True)
        if long_tail_enabled and bool(((cfg.get("long_tail", {}) or {}).get(
            "staged_training", {}) or {}).get("enabled", False))
        else train_df
    )
    tail_replay_df = _tail_replay_frame(train_df, cfg) if long_tail_enabled else None
    train_ds = MultiTaskImageDataset(stage_one_df, transform=train_tf, **common_kwargs)
    val_ds = MultiTaskImageDataset(val_df, transform=eval_tf, **common_kwargs)
    test_ds = MultiTaskImageDataset(
        test_df, transform=eval_tf, **common_kwargs
    )

    batch_size = cfg["training"]["batch_size"]
    configured_workers = int(cfg["training"].get("num_workers", 4))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    num_workers = (
        max(1, configured_workers // world_size)
        if configured_workers > 0
        else 0
    )
    train_loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    eval_loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
        eval_loader_kwargs["prefetch_factor"] = 2

    species_col = target_cols.get("species", cfg["data"]["target_col"])
    train_sampler = _training_sampler(train_ds, stage_one_df, cfg, species_col)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    head_val_loader = None
    head_test_loader = None
    tail_replay_loader = None
    if long_tail_enabled:
        head_val_ds = MultiTaskImageDataset(
            val_df[val_df["__long_tail_group__"] == "head"].reset_index(drop=True),
            transform=eval_tf,
            **common_kwargs,
        )
        head_test_ds = MultiTaskImageDataset(
            test_df[test_df["__long_tail_group__"] == "head"].reset_index(drop=True),
            transform=eval_tf,
            **common_kwargs,
        )
        head_val_loader = DataLoader(head_val_ds, batch_size=batch_size, shuffle=False, **eval_loader_kwargs)
        head_test_loader = DataLoader(head_test_ds, batch_size=batch_size, shuffle=False, **eval_loader_kwargs)
        if tail_replay_df is not None:
            tail_ds = MultiTaskImageDataset(tail_replay_df, transform=train_tf, **common_kwargs)
            tail_sampler = _training_sampler(
                tail_ds, tail_replay_df, cfg, species_col
            )
            tail_replay_loader = DataLoader(
                tail_ds,
                batch_size=batch_size,
                shuffle=tail_sampler is None,
                sampler=tail_sampler,
                **train_loader_kwargs,
            )

    prediction_loaders: dict[str, DataLoader] = {}
    prediction_frames: dict[str, object] = {}
    inference_cfg = cfg.get("inference", {}) or {}
    if (
        cfg.get("data", {}).get("dataset_format") == "fish_pickle"
        and bool(inference_cfg.get("enabled", True))
    ):
        raw_prediction_frames = load_fish_prediction_metadata(cfg)
        for split_name, prediction_df in raw_prediction_frames.items():
            prediction_image_col = cfg["data"]["image_col"]
            prediction_crop = cfg["data"].get("crop_to_foreground", False)
            if cache_enabled:
                prediction_df = _build_cache_safely(cfg, prediction_df)
                prediction_df = prediction_df[
                    prediction_df["_cached_image_path"].notna()
                ].reset_index(drop=True)
                prediction_image_col = "_cached_image_path"
                prediction_crop = False
            prediction_kwargs = {
                "root_dir": common_kwargs["root_dir"],
                "mask_col": common_kwargs["mask_col"],
                "crop_pad": common_kwargs["crop_pad"],
                "image_col": prediction_image_col,
                "crop_to_foreground": prediction_crop,
            }
            dataset = InferenceImageDataset(
                prediction_df,
                transform=eval_tf,
                **prediction_kwargs,
            )
            prediction_loaders[split_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                **eval_loader_kwargs,
            )
            prediction_frames[split_name] = prediction_df

    species_counts = (
        train_df.groupby(species_col)["__species_count__"].first().astype(int).to_dict()
        if long_tail_enabled
        else train_df[species_col].value_counts().astype(int).to_dict()
    )
    cohort_rules = {
        "head_over_10": lambda value: value > 10,
        "tail_10_or_fewer": lambda value: value <= 10,
        "few_shot_2_to_5": lambda value: 2 <= value <= 5,
        "medium_shot_6_to_20": lambda value: 6 <= value <= 20,
        "many_shot_over_20": lambda value: value > 20,
    }
    cohort_summary = {
        name: {
            "n_species": sum(bool(rule(count)) for count in species_counts.values()),
            "n_samples": sum(
                count for count in species_counts.values() if rule(count)
            ),
        }
        for name, rule in cohort_rules.items()
    }

    split_summary = {}
    if profile.loader_mode == "condition":
        split_summary["training_condition"] = input_condition
    split_summary.update(
        {
            "target_cols": target_cols,
            "split_target_col": split_target_col,
            "split_strategy": cfg["split"].get(
                "strategy", "group_stratified"
            ),
            "num_classes_by_task": {
                task: len(label_to_index)
                for task, label_to_index in label_to_index_by_task.items()
            },
            "classes_by_task": {
                task: list(label_to_index.keys())
                for task, label_to_index in label_to_index_by_task.items()
            },
            "labelled_rows_by_task": {
                task: {
                    "train": int(train_df[col].notna().sum()),
                    "val": int(val_df[col].notna().sum()),
                    "test": int(test_df[col].notna().sum()),
                }
                for task, col in target_cols.items()
            },
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_sampling_units": train_df[group_col].nunique(),
            "val_sampling_units": val_df[group_col].nunique(),
            "test_sampling_units": test_df[group_col].nunique(),
            "prediction_rows": {
                name: len(frame) for name, frame in prediction_frames.items()
            },
            "long_tail": {
                "enabled": long_tail_enabled,
                "head_train_rows": int((train_df.get("__long_tail_group__") == "head").sum()) if long_tail_enabled else 0,
                "tail_train_rows": int((train_df.get("__long_tail_group__") == "tail").sum()) if long_tail_enabled else 0,
                "tail_replay_rows": len(tail_replay_df) if tail_replay_df is not None else 0,
                "frequency_cohorts": cohort_summary,
            },
        }
    )

    context = None
    if profile.loader_mode == "condition":
        context = {
            "test_df": test_df,
            "dataset_kwargs": common_kwargs,
            "batch_size": batch_size,
            "loader_kwargs": eval_loader_kwargs,
            "image_size": image_size,
            "preprocessing": preprocessing,
            "augmentation": augmentation,
            "original_colour_retention": 1.0,
            "training_condition": input_condition,
        }

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_to_index_by_task=label_to_index_by_task,
        index_to_label_by_task=index_to_label_by_task,
        split_summary=split_summary,
        train_df=train_df,
        target_cols=target_cols,
        test_loader_context=context,
        prediction_loaders=prediction_loaders,
        prediction_frames=prediction_frames,
        head_val_loader=head_val_loader,
        head_test_loader=head_test_loader,
        tail_replay_loader=tail_replay_loader,
        species_counts=species_counts,
        stage_one_train_df=stage_one_df,
        tail_replay_df=tail_replay_df,
    )


def make_standard_loaders(cfg: dict):
    bundle = make_profile_loaders(cfg, get_profile("standard"))
    return legacy_loader_tuple(bundle)


def legacy_loader_tuple(bundle: LoaderBundle) -> tuple:
    return (
        bundle.train_loader,
        bundle.val_loader,
        bundle.test_loader,
        bundle.label_to_index_by_task,
        bundle.index_to_label_by_task,
        bundle.split_summary,
        bundle.train_df,
        bundle.target_cols,
    )


def legacy_cue_loader_tuple(bundle: LoaderBundle) -> tuple:
    return (*legacy_loader_tuple(bundle), bundle.test_loader_context)
