"""Failure-isolated optional Weights & Biases lifecycle adapter.

The module does not import :mod:`wandb` at import time.  Every backend action
is best-effort because local CSV, JSON, report, and checkpoint files remain the
scientific record.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

from .config import canonical_condition_relation
from .config import canonical_tracking_config
from .config import condition_name
from .config import flatten_slash_config
from .config import identity_summary
from .config import resolved_training_condition
from .tables import CLASSIFICATION_REPORT_COLUMNS
from .tables import canonical_table_records
from .tables import classification_report_rows
from .tables import numeric_metrics
from .tables import robustness_ratio
from .tables import unique_columns
from .tables import valid_confusion_labels


class WandbLogger:
    """Null-capable, exception-safe adapter around one W&B run."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        run_name: str,
        out_dir: str | Path,
        backend: Any = None,
        run: Any = None,
        disabled_reason: str | None = None,
    ) -> None:
        self.cfg = copy.deepcopy(dict(cfg))
        self.run_name = str(run_name)
        self.out_dir = Path(out_dir)
        self.backend = backend
        self._run = run
        self.disabled_reason = disabled_reason
        self.failures: list[str] = []
        self.degraded = False
        self._active = run is not None
        self._finished = False
        self._classification_rows: list[dict[str, Any]] = []
        self._test_conditions: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._run is not None and self._active

    @property
    def run(self) -> Any:
        """Expose the underlying run for temporary compatibility only."""
        return self._run

    def _failure(self, operation: str, exc: Exception) -> None:
        message = f"W&B {operation} failed; continuing locally: {exc}"
        self.failures.append(message)
        self.degraded = True
        self._active = False
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _log(self, payload: Mapping[str, Any]) -> bool:
        if not self.enabled or not payload:
            return False
        try:
            self._run.log(dict(payload))
            return True
        except Exception as exc:
            self._failure("log", exc)
            return False

    def update_summary(self, values: Mapping[str, Any]) -> bool:
        if not self.enabled or not values:
            return False
        try:
            self._run.summary.update(dict(values))
            return True
        except Exception as exc:
            self._failure("summary update", exc)
            return False

    def log_epoch_metrics(
        self,
        *,
        epoch: int,
        learning_rate: float,
        train_metrics: Mapping[str, Any],
        val_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "learning_rate": float(learning_rate),
        }
        payload.update(numeric_metrics("train", train_metrics))
        payload.update(numeric_metrics("val", val_metrics))
        self._log(payload)
        return payload

    def update_best(
        self, *, best_epoch: int, best_val_score: float, selection_metric: str
    ) -> dict[str, Any]:
        values = {
            "best_epoch": int(best_epoch),
            "best_val_score": float(best_val_score),
            "selection_metric": str(selection_metric),
        }
        self.update_summary(values)
        return values

    def log_test_condition(
        self,
        condition: str | Mapping[str, Any],
        metrics: Mapping[str, Any],
        *,
        train_condition: str | Mapping[str, Any] | None = None,
        condition_relation: str | None = None,
        preserve_legacy_alias: bool | None = None,
        update_summary: bool = True,
    ) -> dict[str, Any]:
        """Log one test condition without creating another W&B run."""
        test_name = condition_name(condition)
        train_name = condition_name(
            train_condition
            if train_condition is not None
            else resolved_training_condition(self.cfg)
        )
        relation = condition_relation or canonical_condition_relation(
            train_name, test_name
        )
        self._test_conditions.add(test_name)
        payload: dict[str, Any] = {
            "train_condition": train_name,
            "test_condition": test_name,
            "condition_relation": relation,
            **numeric_metrics(f"test/{test_name}", metrics),
        }
        if preserve_legacy_alias is None:
            preserve_legacy_alias = test_name == "original"
        if preserve_legacy_alias:
            payload.update(numeric_metrics("test", metrics))
        self._log(payload)

        if update_summary and test_name == "original":
            summary = dict(numeric_metrics("test", metrics))
            for task in ("genus", "species"):
                key = f"{task}_macro_f1"
                if key in metrics:
                    summary[f"test_original_{key}"] = metrics[key]
            if "mean_macro_f1" in metrics:
                summary["test_original_mean_macro_f1"] = metrics[
                    "mean_macro_f1"
                ]
            summary.update({
                "train_condition": train_name,
                "test_condition": test_name,
                "condition_relation": relation,
            })
            self.update_summary(summary)
        return payload

    def _confusion_settings(
        self,
    ) -> tuple[bool, set[str] | None, set[str] | None]:
        wandb_cfg = self.cfg.get("wandb", {}) or {}
        settings = (
            wandb_cfg.get("confusion_matrices", {})
            if isinstance(wandb_cfg, Mapping)
            else {}
        )
        if not isinstance(settings, Mapping):
            settings = {}
        enabled = bool(settings.get("enabled", True))
        conditions = settings.get("conditions", ["original"])
        tasks = settings.get("tasks")
        condition_set = (
            None
            if conditions is None
            else {str(value) for value in conditions}
        )
        task_set = (
            None if tasks is None else {str(value) for value in tasks}
        )
        return enabled, condition_set, task_set

    def should_log_confusion_matrix(self, condition: str, task: str) -> bool:
        enabled, conditions, tasks = self._confusion_settings()
        return (
            enabled
            and (conditions is None or str(condition) in conditions)
            and (tasks is None or str(task) in tasks)
        )

    def log_confusion_matrix(
        self,
        *,
        condition: str | Mapping[str, Any],
        task: str,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        class_names: Sequence[str],
        preserve_legacy_alias: bool | None = None,
        title: str | None = None,
    ) -> str | None:
        condition_value = condition_name(condition)
        if not self.enabled or not self.should_log_confusion_matrix(
            condition_value, task
        ):
            return None
        names = [str(name) for name in class_names]
        true_labels, predicted_labels = valid_confusion_labels(
            y_true, y_pred, len(names)
        )
        if not true_labels:
            return None
        try:
            plot = self.backend.plot.confusion_matrix(
                y_true=true_labels,
                preds=predicted_labels,
                class_names=names,
                title=title
                or f"Confusion Matrix ({condition_value}: {task})",
            )
        except Exception as exc:
            self._failure("confusion matrix", exc)
            return None
        key = f"confusion_matrix/{condition_value}/{task}"
        payload = {key: plot}
        if preserve_legacy_alias is None:
            preserve_legacy_alias = condition_value == "original"
        if preserve_legacy_alias:
            payload[f"confusion_matrix_{task}"] = plot
        self._log(payload)
        return key

    def _table(
        self, rows: Any, columns: Sequence[str] | None = None
    ) -> Any:
        records = canonical_table_records(rows)
        selected = unique_columns(records, columns)
        data = [[row.get(column) for column in selected] for row in records]
        return self.backend.Table(columns=selected, data=data)

    def log_classification_report(
        self,
        *,
        condition: str | Mapping[str, Any],
        task: str,
        report: Any,
        metrics: Mapping[str, Any] | None = None,
        train_condition: str | Mapping[str, Any] | None = None,
        condition_relation: str | None = None,
    ) -> list[dict[str, Any]]:
        test_name = condition_name(condition)
        train_name = condition_name(
            train_condition
            if train_condition is not None
            else resolved_training_condition(self.cfg)
        )
        relation = condition_relation or canonical_condition_relation(
            train_name, test_name
        )
        model = self.cfg.get("model", {}) or {}
        rows = classification_report_rows(
            report,
            model=model.get("name") if isinstance(model, Mapping) else None,
            task=str(task),
            train_condition=train_name,
            test_condition=test_name,
            condition_relation=relation,
            metrics=metrics,
        )
        if not rows or not self.enabled:
            return rows
        self._classification_rows.extend(rows)
        try:
            payload = {
                "tables/classification_report_by_condition": self._table(
                    self._classification_rows,
                    CLASSIFICATION_REPORT_COLUMNS,
                )
            }
            if test_name == "original":
                payload["tables/classification_report_original"] = (
                    self._table(rows, CLASSIFICATION_REPORT_COLUMNS)
                )
            self._log(payload)
        except Exception as exc:
            self._failure("classification report", exc)
        return rows

    def log_test_metrics_table(self, rows: Any) -> bool:
        if not self.enabled:
            return False
        try:
            table = self._table(rows)
            return self._log({"tables/test_metrics_by_condition": table})
        except Exception as exc:
            self._failure("test metrics table", exc)
            return False

    def log_robustness_table(
        self, rows: Any, *, transform_summary: Any | None = None
    ) -> dict[str, Any]:
        records = canonical_table_records(rows)
        scalar_payload: dict[str, Any] = {}
        for row in records:
            condition = str(row.get("test_condition") or "original")
            task = str(row.get("task") or "mean")
            ratio = row.get("ratio_to_original")
            if ratio is None:
                ratio = robustness_ratio(
                    row.get("macro_f1"), row.get("original_macro_f1")
                )
                row["ratio_to_original"] = ratio
            try:
                ratio_value = float(ratio)
            except (TypeError, ValueError):
                ratio_value = float("nan")
            relative_drop = row.get("relative_drop")
            if relative_drop is None:
                relative_drop = (
                    1.0 - ratio_value
                    if math.isfinite(ratio_value)
                    else float("nan")
                )
                row["relative_drop"] = relative_drop
            scalar_payload[f"robustness/{condition}/{task}_ratio"] = (
                ratio_value
            )
            scalar_payload[
                f"robustness/{condition}/{task}_relative_drop"
            ] = relative_drop

        if not self.enabled:
            return scalar_payload
        try:
            robustness_table = self._table(records)
            payload: dict[str, Any] = {
                **scalar_payload,
                "tables/robustness_summary": robustness_table,
                "cue_suppression/macro_f1_ratios": robustness_table,
            }
            if transform_summary is not None:
                payload["cue_suppression/transform_summary"] = self._table(
                    transform_summary
                )
            self._log(payload)
        except Exception as exc:
            self._failure("robustness table", exc)
        return scalar_payload

    def log_comparison_table(self, rows: Any) -> bool:
        if not self.enabled:
            return False
        try:
            records = canonical_table_records(rows)
            payload: dict[str, Any] = {
                "tables/matched_vs_rgb_stress": self._table(records)
            }
            for row in records:
                condition = str(row.get("test_condition"))
                task = str(row.get("task") or "mean")
                if row.get("adaptation_gain") is not None:
                    payload[
                        f"comparison/{condition}/{task}_adaptation_gain"
                    ] = row["adaptation_gain"]
            return self._log(payload)
        except Exception as exc:
            self._failure("comparison table", exc)
            return False

    def alert(self, *, title: str, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            self._run.alert(title=title, text=text)
            return True
        except Exception as exc:
            self._failure("alert", exc)
            return False

    def log_artifacts(
        self,
        paths: Mapping[str, str | Path] | Iterable[str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if not self.enabled:
            return []
        if isinstance(paths, Mapping):
            named_paths = [
                (str(name), Path(path)) for name, path in paths.items()
            ]
        else:
            named_paths = [
                (Path(path).name, Path(path)) for path in paths
            ]
        existing = [
            (name, path) for name, path in named_paths if path.is_file()
        ]
        model_paths = [
            item
            for item in existing
            if item[0] == "best_model.pt"
            or item[1].name == "best_model.pt"
        ]
        lightweight = [item for item in existing if item not in model_paths]
        base_metadata = self._artifact_metadata()
        scientific_metadata = {**base_metadata, **dict(metadata or {})}
        model_artifact_metadata = {
            **scientific_metadata,
            **dict(model_metadata or {}),
        }
        logged: list[str] = []
        try:
            if lightweight:
                artifact_name = f"{self.run_name}-scientific-record"
                artifact = self.backend.Artifact(
                    name=artifact_name,
                    type="scientific-results",
                    metadata=scientific_metadata,
                )
                for _, path in lightweight:
                    artifact.add_file(str(path))
                self._run.log_artifact(artifact)
                logged.append(artifact_name)

            wandb_cfg = self.cfg.get("wandb", {}) or {}
            log_model = False
            if log_model and model_paths:
                artifact_name = f"{self.run_name}-best-model"
                artifact = self.backend.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata=model_artifact_metadata,
                )
                for _, path in model_paths:
                    artifact.add_file(str(path))
                self._run.log_artifact(artifact)
                logged.append(artifact_name)
        except Exception as exc:
            self._failure("artifact logging", exc)
        return logged

    def _artifact_metadata(self) -> dict[str, Any]:
        condition = resolved_training_condition(self.cfg)
        model = self.cfg.get("model", {}) or {}
        data = self.cfg.get("data", {}) or {}
        preprocessing = self.cfg.get("preprocessing", {}) or {}
        tasks = data.get("target_cols", {}) if isinstance(data, Mapping) else {}
        runtime = self.cfg.get("runtime", {}) or {}
        return {
            "architecture": (
                model.get("name") if isinstance(model, Mapping) else None
            ),
            "tasks": list(tasks) if isinstance(tasks, Mapping) else [],
            "image_size": (
                preprocessing.get("image_size")
                if isinstance(preprocessing, Mapping)
                and preprocessing.get("image_size") is not None
                else data.get("image_size")
                if isinstance(data, Mapping)
                else None
            ),
            "training_condition": condition["name"],
            "seed": self.cfg.get("seed"),
            "git_commit": (
                runtime.get("git_commit")
                if isinstance(runtime, Mapping)
                else None
            )
            or os.getenv("GIT_COMMIT"),
        }

    def finalise_run(
        self,
        *,
        status: str = "completed",
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        if self._run is None or self._finished:
            return
        if self._active:
            values = dict(summary or {})
            values["run_status"] = str(status)
            if self._test_conditions:
                values.setdefault(
                    "number_of_test_conditions", len(self._test_conditions)
                )
            self.update_summary(values)
        try:
            self._run.finish()
        except Exception as exc:
            message = f"W&B finish failed; continuing locally: {exc}"
            self.failures.append(message)
            self.degraded = True
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        finally:
            self._active = False
            self._finished = True


def _enabled_for_profile(cfg: Mapping[str, Any], profile: Any) -> bool:
    wandb_cfg = cfg.get("wandb", {}) or {}
    enabled = isinstance(wandb_cfg, Mapping) and bool(
        wandb_cfg.get("enabled", False)
    )
    if profile is not None and hasattr(profile, "wandb"):
        enabled = enabled and bool(profile.wandb)
    return enabled


def create_wandb_logger(
    cfg: Mapping[str, Any],
    run_name: str,
    out_dir: str | Path,
    profile: Any = None,
    *,
    backend: Any = None,
) -> WandbLogger:
    """Create one real or null logger without exposing backend failures."""
    if not _enabled_for_profile(cfg, profile):
        return WandbLogger(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            disabled_reason="disabled",
        )

    wandb_cfg = cfg.get("wandb", {}) or {}
    mode = (
        wandb_cfg.get("mode")
        if isinstance(wandb_cfg, Mapping) and wandb_cfg.get("mode")
        else os.getenv("WANDB_MODE") or "online"
    )
    if str(mode).lower() == "disabled":
        return WandbLogger(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            disabled_reason="mode=disabled",
        )

    if backend is None:
        try:
            backend = importlib.import_module("wandb")
        except Exception as exc:
            warnings.warn(
                f"W&B is unavailable; continuing locally: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return WandbLogger(
                cfg=cfg,
                run_name=run_name,
                out_dir=out_dir,
                disabled_reason="unavailable",
            )

    project = (
        wandb_cfg.get("project")
        if isinstance(wandb_cfg, Mapping)
        else None
    ) or os.getenv("WANDB_PROJECT") or "fish-species"
    entity = (
        wandb_cfg.get("entity")
        if isinstance(wandb_cfg, Mapping)
        else None
    ) or os.getenv("WANDB_ENTITY") or None
    configured_name = str(wandb_cfg.get("name")) + "_" + str(run_name)
    group = (
        wandb_cfg.get("group")
        if isinstance(wandb_cfg, Mapping)
        else None
    ) or os.getenv("WANDB_RUN_GROUP") or None
    job_type = (
        wandb_cfg.get("job_type", "train")
        if isinstance(wandb_cfg, Mapping)
        else "train"
    )
    tags = (
        list(wandb_cfg.get("tags", []) or [])
        if isinstance(wandb_cfg, Mapping)
        else []
    )
    if os.getenv("SLURM_JOB_ID") and "slurm" not in tags:
        tags.append("slurm")

    runtime = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "hostname": os.getenv("HOSTNAME"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    tracking_config = canonical_tracking_config(cfg, runtime)
    run = None
    try:
        run = backend.init(
            project=project,
            entity=entity,
            name=configured_name,
            group=group,
            job_type=job_type,
            tags=tags,
            config=tracking_config,
            dir=str(out_dir),
            mode=mode,
            save_code=(
                bool(wandb_cfg.get("save_code", True))
                if isinstance(wandb_cfg, Mapping)
                else True
            ),
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("learning_rate", step_metric="epoch")
    except Exception as exc:
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass
        warnings.warn(
            f"W&B initialisation failed; continuing locally: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return WandbLogger(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            backend=backend,
            disabled_reason="initialisation failed",
        )

    logger = WandbLogger(
        cfg=cfg,
        run_name=run_name,
        out_dir=out_dir,
        backend=backend,
        run=run,
    )
    logger.update_summary(
        identity_summary(cfg, run_name=str(configured_name))
    )
    return logger


__all__ = [
    "CLASSIFICATION_REPORT_COLUMNS",
    "WandbLogger",
    "canonical_condition_relation",
    "create_wandb_logger",
    "flatten_slash_config",
    "robustness_ratio",
]
