"""Fixed-RGB stress-test condition generation and loader construction.

This module never generates matched-condition training runs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ..training.checkpoints import load_checkpoint
from ..training.checkpoints import load_model_state_compat
from ..data.datasets import MultiTaskFishImageDataset
from ..data.transforms import build_split_transform
from ..results.writing import save_json
from ..training.epochs import run_hierarchy_epoch as run_epoch


def _inclusive_float_sequence(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive, rounded sequence in either direction."""
    start = float(start)
    stop = float(stop)
    step = abs(float(step))
    if step == 0:
        raise ValueError("Sequence step must be greater than zero.")

    direction = -1.0 if start > stop else 1.0
    values = []
    current = start
    tolerance = step * 1e-6
    if direction < 0:
        while current >= stop - tolerance:
            values.append(round(current, 10))
            current -= step
    else:
        while current <= stop + tolerance:
            values.append(round(current, 10))
            current += step

    if not values or not math.isclose(values[-1], stop, abs_tol=tolerance):
        values.append(stop)
    return values


def _runtime_condition(raw: dict) -> dict:
    condition = {
        "condition": str(raw.get("name") or raw.get("condition")),
        "feature": str(raw.get("feature", "baseline")),
        "transform": str(raw.get("transform", "original")),
        "strength": raw.get("strength", 0.0),
    }
    parameters = raw.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise TypeError("condition parameters must be a mapping")
    condition.update(parameters)
    if condition["transform"] == "grayscale":
        condition.setdefault("retention", 0.0)
    return condition


def _canonical_test_conditions(cfg: dict) -> list[dict] | None:
    evaluation = cfg.get("evaluation")
    if not isinstance(evaluation, dict) or "test_conditions" not in evaluation:
        return None
    schedule = evaluation.get("test_conditions", {}) or {}
    if not isinstance(schedule, dict):
        raise TypeError("evaluation.test_conditions must be a mapping")
    legacy = cfg.get("test_cue_suppression", {}) or {}
    legacy_enabled = isinstance(legacy, dict) and bool(legacy.get("enabled", False))
    if not bool(schedule.get("enabled", False)) and not legacy_enabled:
        return []

    # Transitional resolved run specs may still carry the historical boolean
    # override while their condition catalogue already lives under evaluation.
    # In that case the alias enables the canonical catalogue; it does not
    # regenerate a second legacy condition sequence.
    if not schedule.get("conditions") and legacy_enabled:
        return None

    configured = schedule.get("conditions", [])
    if not isinstance(configured, list) or not configured:
        raise ValueError(
            "evaluation.test_conditions.conditions must be a non-empty list"
        )
    sweep = cfg.get("sweep", {}) or {}
    registry: dict[str, dict] = {}
    if isinstance(sweep, dict) and sweep.get("conditions"):
        from ..config.normalization import normalize_conditions

        registry.update({
            str(item["name"]): item
            for item in normalize_conditions(sweep["conditions"])
        })
    registry.setdefault(
        "original",
        {
            "name": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
            "parameters": {},
        },
    )

    resolved = []
    for index, item in enumerate(configured):
        if isinstance(item, str):
            try:
                item = registry[item]
            except KeyError as exc:
                raise ValueError(
                    "Unknown evaluation.test_conditions condition "
                    f"{item!r} at index {index}"
                ) from exc
        if not isinstance(item, dict):
            raise TypeError(
                "evaluation.test_conditions.conditions entries must be names "
                "or complete condition mappings"
            )
        resolved.append(_runtime_condition(item))
    names = [item["condition"] for item in resolved]
    if len(names) != len(set(names)):
        raise ValueError(
            "evaluation.test_conditions.conditions contains duplicate names"
        )
    return resolved


def generate_test_cue_conditions(cfg: dict) -> list[dict]:
    """Create deterministic test conditions from ``test_cue_suppression``."""
    canonical = _canonical_test_conditions(cfg)
    if canonical is not None:
        return canonical
    cue_cfg = cfg.get("test_cue_suppression", {}) or {}
    if not bool(cue_cfg.get("enabled", False)):
        return []

    conditions: list[dict] = []

    saturation_cfg = cue_cfg.get("saturation", {}) or {}
    if bool(saturation_cfg.get("enabled", True)):
        values = saturation_cfg.get("values")
        if values is None:
            values = _inclusive_float_sequence(
                saturation_cfg.get("start", 1.0),
                saturation_cfg.get("stop", 0.0),
                saturation_cfg.get("step", 0.01),
            )
        for retention in values:
            retention = float(retention)
            if not 0.0 <= retention <= 1.0:
                raise ValueError(
                    f"Saturation retention values must be in [0, 1], got {retention}."
                )
            percentage = int(round(retention * 100))
            conditions.append({
                "condition": f"saturation_{percentage:03d}pct",
                "feature": "colour",
                "transform": "saturation",
                "strength": round(float(1.0 - retention), 10),
                "retention": retention,
            })

    grayscale_cfg = cue_cfg.get("grayscale", {}) or {}
    if bool(grayscale_cfg.get("enabled", True)):
        conditions.append({
            "condition": "grayscale",
            "feature": "colour",
            "transform": "grayscale",
            "strength": 1.0,
            "retention": 0.0,
        })

    channel_cfg = cue_cfg.get("channel_shuffle", {}) or {}
    if bool(channel_cfg.get("enabled", True)):
        orders = channel_cfg.get("orders", [[2, 0, 1]])
        for order in orders:
            order = [int(i) for i in order]
            conditions.append({
                "condition": "channel_shuffle_" + "".join(str(i) for i in order),
                "feature": "colour",
                "transform": "channel_shuffle",
                "strength": 1.0,
                "order": order,
            })

    bilateral_cfg = cue_cfg.get("bilateral_filter", {}) or {}
    if bool(bilateral_cfg.get("enabled", True)):
        settings = bilateral_cfg.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        for setting in settings:
            diameter = int(setting["diameter"])
            sigma_colour = float(setting["sigma_colour"])
            sigma_space = float(setting["sigma_space"])
            conditions.append({
                "condition": (
                    f"bilateral_d{diameter}_c{sigma_colour:g}_s{sigma_space:g}"
                ),
                "feature": "texture",
                "transform": "bilateral_filter",
                "strength": sigma_colour,
                "diameter": diameter,
                "sigma_colour": sigma_colour,
                "sigma_space": sigma_space,
            })

    gaussian_cfg = cue_cfg.get("gaussian_blur", {}) or {}
    if bool(gaussian_cfg.get("enabled", True)):
        for sigma in gaussian_cfg.get("sigmas", [0.5, 1.0, 2.0, 4.0]):
            sigma = float(sigma)
            conditions.append({
                "condition": f"gaussian_sigma_{sigma:g}",
                "feature": "texture",
                "transform": "gaussian_blur",
                "strength": sigma,
                "sigma": sigma,
            })

    patch_cfg = cue_cfg.get("patch_shuffle", {}) or {}
    if bool(patch_cfg.get("enabled", True)):
        seed = int(patch_cfg.get("seed", cfg.get("seed", 0)))
        for grid_size in patch_cfg.get("grid_sizes", [2, 4, 8]):
            grid_size = int(grid_size)
            conditions.append({
                "condition": f"patch_shuffle_grid_{grid_size}",
                "feature": "shape",
                "transform": "patch_shuffle",
                "strength": grid_size,
                "grid_size": grid_size,
                "seed": seed,
            })

    names = [condition["condition"] for condition in conditions]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate test cue condition names: {duplicates}")

    requested_names = cue_cfg.get("condition_names")
    if requested_names is not None:
        if not isinstance(requested_names, list) or not requested_names:
            raise ValueError(
                "test_cue_suppression.condition_names must be a non-empty list"
            )
        if any(
            not isinstance(name, str) or not name.strip()
            for name in requested_names
        ):
            raise ValueError(
                "test_cue_suppression.condition_names must contain non-empty "
                "condition-name strings"
            )
        duplicate_requests = sorted({
            name for name in requested_names if requested_names.count(name) > 1
        })
        if duplicate_requests:
            raise ValueError(
                "Duplicate test_cue_suppression.condition_names: "
                f"{duplicate_requests}"
            )
        available_names = set(names)
        unknown = sorted(set(requested_names) - available_names)
        if unknown:
            raise ValueError(
                "Unknown test_cue_suppression.condition_names: "
                f"{unknown}; available conditions: {names}"
            )
        requested = set(requested_names)
        conditions = [
            condition
            for condition in conditions
            if condition["condition"] in requested
        ]
    return conditions


def _test_condition_signature(condition: dict, original_colour_retention: float) -> str:
    """Identify conditions that produce exactly the same transformed input."""
    transform_name = condition["transform"]
    if transform_name == "original":
        return f"colour_retention:{original_colour_retention:.10f}"
    if transform_name == "saturation":
        return f"colour_retention:{float(condition['retention']):.10f}"
    if transform_name == "grayscale":
        return "colour_retention:0.0000000000"
    return json.dumps(condition, sort_keys=True)


def make_test_condition_loader(test_loader_context: dict, condition: dict) -> DataLoader:
    preprocessing = test_loader_context.get("preprocessing") or {
        "image_size": int(test_loader_context["image_size"])
    }
    transform = build_split_transform(
        split="test",
        preprocessing=preprocessing,
        condition=condition,
        original_colour_retention=float(
            test_loader_context["original_colour_retention"]
        ),
    )
    dataset = MultiTaskFishImageDataset(
        test_loader_context["test_df"],
        transform=transform,
        **test_loader_context["dataset_kwargs"],
    )
    return DataLoader(
        dataset,
        batch_size=int(test_loader_context["batch_size"]),
        shuffle=False,
        **test_loader_context["loader_kwargs"],
    )


def evaluate_test_cue_suppression(
    *,
    cfg: dict,
    run_name: str,
    out_dir: Path,
    model: nn.Module,
    checkpoint_name: str,
    checkpoint_path: Path,
    baseline_metrics: dict,
    test_loader_context: dict,
    criteria: dict[str, nn.Module],
    target_cols: dict[str, str],
    device: torch.device,
    use_amp: bool,
    task_loss_weights: dict[str, float],
    normalize_loss_by_active_tasks: bool,
    hierarchy_cfg: dict,
    child_to_parent_matrix: torch.Tensor | None,
    metric_context: dict | None = None,
    wandb_logger=None,
) -> dict:
    """Evaluate one fixed checkpoint under all configured test manipulations."""
    conditions = generate_test_cue_conditions(cfg)
    if not conditions:
        return {
            "enabled": False,
            "checkpoint": checkpoint_name,
            "n_conditions": 0,
        }

    cue_dir = out_dir / f"{checkpoint_name}_cue_suppression"
    cue_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "checkpoint": checkpoint_name,
            "checkpoint_path": str(checkpoint_path),
            "configuration": cfg.get("test_cue_suppression", {}) or {},
        },
        cue_dir / "cue_suppression_config.json",
    )

    original_colour_retention = float(
        test_loader_context["original_colour_retention"]
    )
    original_condition = {
        "condition": "original",
        "feature": "baseline",
        "transform": "original",
        "strength": 0.0,
        "retention": original_colour_retention,
    }

    metric_cache = {
        _test_condition_signature(
            original_condition,
            original_colour_retention,
        ): baseline_metrics
    }

    condition_metric_rows = []
    ratio_rows = []

    def record_condition(
        condition: dict,
        metrics: dict,
        reused: bool,
    ) -> None:
        parameters = json.dumps(
            {
                key: value
                for key, value in condition.items()
                if key not in {
                    "condition",
                    "feature",
                    "transform",
                    "strength",
                }
            },
            sort_keys=True,
        )

        condition_metric_rows.append({
            "run_name": run_name,
            "checkpoint": checkpoint_name,
            "model": cfg.get("model", {}).get("name"),
            "condition": condition["condition"],
            "feature": condition["feature"],
            "transform": condition["transform"],
            "strength": condition.get("strength"),
            "parameters": parameters,
            "reused_identical_evaluation": bool(reused),
            **metrics,
        })

        for task in target_cols:
            metric_key = f"{task}_macro_f1"
            transformed_score = float(
                metrics.get(metric_key, float("nan"))
            )
            original_score = float(
                baseline_metrics.get(metric_key, float("nan"))
            )

            if (
                math.isnan(transformed_score)
                or math.isnan(original_score)
                or original_score == 0.0
            ):
                ratio = float("nan")
            else:
                ratio = transformed_score / original_score

            ratio_rows.append({
                "run_name": run_name,
                "checkpoint": checkpoint_name,
                "model": cfg.get("model", {}).get("name"),
                "task": task,
                "condition": condition["condition"],
                "feature": condition["feature"],
                "transform": condition["transform"],
                "strength": condition.get("strength"),
                "parameters": parameters,
                "n": metrics.get(f"{task}_n"),
                "macro_f1": transformed_score,
                "original_macro_f1": original_score,
                "ratio_to_original": ratio,
                "relative_drop": (
                    1.0 - ratio
                    if not math.isnan(ratio)
                    else float("nan")
                ),
            })

    record_condition(
        original_condition,
        baseline_metrics,
        reused=True,
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    missing, ignored = load_model_state_compat(model, checkpoint["model_state"])
    if missing:
        print("Newly initialized checkpoint modules: " + ", ".join(missing))
    if ignored:
        print("Checkpoint parameters not used by this architecture: " + ", ".join(ignored))

    for condition_index, condition in enumerate(conditions, start=1):
        signature = _test_condition_signature(
            condition,
            original_colour_retention,
        )
        reused = signature in metric_cache

        if reused:
            metrics = metric_cache[signature]
            print(
                f"[{checkpoint_name}] Cue test "
                f"{condition_index}/{len(conditions)}: "
                f"{condition['condition']} reuses an identical evaluation."
            )
        else:
            print(
                f"[{checkpoint_name}] Cue test "
                f"{condition_index}/{len(conditions)}: "
                f"{condition['condition']}"
            )

            condition_loader = make_test_condition_loader(
                test_loader_context,
                condition,
            )
            metrics, _, _ = run_epoch(
                model=model,
                loader=condition_loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=(
                    normalize_loss_by_active_tasks
                ),
                hierarchy_cfg=hierarchy_cfg,
                child_to_parent_matrix=child_to_parent_matrix,
                metric_context=metric_context,
            )
            metric_cache[signature] = metrics

        record_condition(
            condition,
            metrics,
            reused=reused,
        )

    condition_metrics_df = pd.DataFrame(condition_metric_rows)
    ratios_df = pd.DataFrame(ratio_rows)

    condition_metrics_path = cue_dir / "test_condition_metrics.csv"
    ratios_path = cue_dir / "macro_f1_ratios.csv"

    condition_metrics_df.to_csv(condition_metrics_path, index=False)
    ratios_df.to_csv(ratios_path, index=False)

    feature_summary = (
        ratios_df[ratios_df["condition"] != "original"]
        .groupby(
            [
                "checkpoint",
                "model",
                "task",
                "feature",
                "transform",
            ],
            dropna=False,
        )
        .agg(
            mean_ratio_to_original=("ratio_to_original", "mean"),
            minimum_ratio_to_original=("ratio_to_original", "min"),
            mean_relative_drop=("relative_drop", "mean"),
            n_conditions=("condition", "count"),
        )
        .reset_index()
    )

    feature_summary_path = cue_dir / "transform_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    if wandb_logger is not None:
        wandb_logger.log_test_metrics_table(condition_metrics_df)
        wandb_logger.log_robustness_table(
            ratios_df,
            transform_summary=feature_summary,
        )

        identity_columns = {
            "run_name",
            "checkpoint",
            "model",
            "condition",
            "feature",
            "transform",
            "strength",
            "parameters",
            "reused_identical_evaluation",
        }

        for row in condition_metric_rows:
            condition_identifier = (
                f"{checkpoint_name}/{row['condition']}"
            )

            wandb_logger.log_test_condition(
                condition_identifier,
                {
                    key: value
                    for key, value in row.items()
                    if key not in identity_columns
                },
                train_condition="original",
                update_summary=False,
            )

    print(
        f"Saved {checkpoint_name} cue-suppression metrics to {cue_dir}"
    )

    return {
        "enabled": True,
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "n_conditions": int(len(condition_metric_rows)),
        "n_unique_evaluations": int(len(metric_cache)),
        "condition_metrics_path": str(condition_metrics_path),
        "macro_f1_ratios_path": str(ratios_path),
        "transform_summary_path": str(feature_summary_path),
    }
