from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONDITION_MATRIX_CONDITION_OUTPUT = "condition_matrix_evaluations.csv"
CONDITION_MATRIX_TASK_OUTPUT = "condition_matrix_task_metrics.csv"
CONDITION_MATRIX_SUMMARY_OUTPUT = "condition_matrix_collection_summary.json"
_CONDITION_REQUIRED = (
    "schema_version",
    "run_name",
    "model",
    "train_condition",
    "test_condition",
    "evaluation_relation",
)
_TASK_REQUIRED = (*_CONDITION_REQUIRED, "task", "macro_f1")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"Skipping unreadable JSON {path}: {exc}")
        return None


def collect_nested_csv(root: Path, relative_path: str, output_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(root.rglob(relative_path)):
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Skipping unreadable CSV {csv_path}: {exc}")
            continue
        frame["source_path"] = str(csv_path.relative_to(root))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    output_path = root / output_name
    combined.to_csv(output_path, index=False)
    print(f"Wrote {len(combined)} rows to {output_path}")
    return combined


def matched_results_long(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in results.iterrows():
        for column in results.columns:
            prefix = "test_"
            suffix = "_macro_f1"
            if not column.startswith(prefix) or not column.endswith(suffix):
                continue
            task = column[len(prefix):-len(suffix)]
            value = row.get(column)
            try:
                macro_f1 = float(value)
            except (TypeError, ValueError):
                macro_f1 = float("nan")
            rows.append({
                "run_name": row.get("run_name"),
                "model": row.get("model"),
                "task": task,
                "train_condition": row.get("train_condition", "original"),
                "train_feature": row.get("train_feature", "baseline"),
                "train_transform": row.get("train_transform", "original"),
                "train_strength": row.get("train_strength"),
                "matched_test_macro_f1": macro_f1,
                "best_epoch": row.get("best_epoch"),
                "best_val_score": row.get("best_val_score"),
                "selection_metric": row.get("selection_metric"),
                "out_dir": row.get("out_dir"),
                "summary_path": row.get("summary_path"),
            })
    return pd.DataFrame(rows)


def add_equivalent_condition_aliases(matched_long: pd.DataFrame) -> pd.DataFrame:
    if matched_long.empty:
        return matched_long
    frames = [matched_long.copy()]
    original = matched_long[matched_long["train_condition"] == "original"].copy()
    if not original.empty:
        original["train_condition"] = "saturation_100pct"
        original["condition_alias_of"] = "original"
        frames.append(original)
    grayscale = matched_long[matched_long["train_condition"] == "grayscale"].copy()
    if not grayscale.empty:
        grayscale["train_condition"] = "saturation_000pct"
        grayscale["condition_alias_of"] = "grayscale"
        frames.append(grayscale)
    combined = pd.concat(frames, ignore_index=True)
    if "condition_alias_of" not in combined:
        combined["condition_alias_of"] = pd.NA
    return combined


def build_comparison(matched_long: pd.DataFrame, cue_ratios: pd.DataFrame) -> pd.DataFrame:
    if matched_long.empty or cue_ratios.empty:
        return pd.DataFrame()
    cue_columns = [
        "model", "task", "condition", "feature", "transform", "strength",
        "macro_f1", "original_macro_f1", "ratio_to_original", "relative_drop",
    ]
    cue = cue_ratios[[column for column in cue_columns if column in cue_ratios.columns]].copy()
    cue = cue.rename(columns={
        "condition": "train_condition",
        "feature": "test_feature",
        "transform": "test_transform",
        "strength": "test_strength",
        "macro_f1": "rgb_model_test_macro_f1",
        "original_macro_f1": "rgb_original_macro_f1",
        "ratio_to_original": "rgb_ratio_to_original",
        "relative_drop": "rgb_relative_drop",
    })
    comparison = add_equivalent_condition_aliases(matched_long).merge(
        cue,
        on=["model", "task", "train_condition"],
        how="inner",
        validate="many_to_one",
    )
    comparison["adaptation_gain_macro_f1"] = (
        comparison["matched_test_macro_f1"] - comparison["rgb_model_test_macro_f1"]
    )
    comparison["matched_ratio_to_rgb_original"] = comparison.apply(
        lambda row: (
            row["matched_test_macro_f1"] / row["rgb_original_macro_f1"]
            if pd.notna(row["matched_test_macro_f1"])
            and pd.notna(row["rgb_original_macro_f1"])
            and float(row["rgb_original_macro_f1"]) != 0.0
            else float("nan")
        ),
        axis=1,
    )
    comparison["matched_relative_drop_from_rgb_original"] = (
        1.0 - comparison["matched_ratio_to_rgb_original"]
    )
    return comparison.sort_values(
        ["model", "task", "test_feature", "test_transform", "test_strength"],
        na_position="first",
    ).reset_index(drop=True)


def _matrix_csv(
    path: Path,
    required_columns: tuple[str, ...],
    root: Path,
    warnings: list[dict[str, str]],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warnings.append({
            "code": "malformed_matrix_csv",
            "path": str(path.relative_to(root)),
            "message": str(exc),
        })
        return pd.DataFrame()
    missing = [column for column in required_columns if column not in frame]
    if missing:
        warnings.append({
            "code": "invalid_matrix_schema",
            "path": str(path.relative_to(root)),
            "message": f"missing required columns: {missing}",
        })
        return pd.DataFrame()
    valid = frame.loc[:, list(required_columns)].notna().all(axis=1)
    invalid_count = int((~valid).sum())
    if invalid_count:
        warnings.append({
            "code": "invalid_matrix_rows",
            "path": str(path.relative_to(root)),
            "message": f"ignored {invalid_count} rows with missing identity fields",
        })
        frame = frame.loc[valid].copy()
    frame["source_path"] = str(path.relative_to(root))
    return frame


def _matrix_manifest(
    path: Path, root: Path, warnings: list[dict[str, str]]
) -> dict | None:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("manifest must contain a JSON object")
        return value
    except Exception as exc:
        warnings.append({
            "code": "malformed_matrix_manifest",
            "path": str(path.relative_to(root)),
            "message": str(exc),
        })
        return None


def _deduplicate_matrix_rows(
    frame: pd.DataFrame,
    keys: list[str],
    warnings: list[dict[str, str]],
    table: str,
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    duplicated = frame.duplicated(subset=keys, keep="first")
    count = int(duplicated.sum())
    if count:
        warnings.append({
            "code": "duplicate_matrix_rows",
            "path": table,
            "message": f"ignored {count} duplicate rows by {keys}",
        })
        frame = frame.loc[~duplicated].copy()
    return frame.reset_index(drop=True), count


def collect_condition_matrix(root: Path) -> dict | None:
    """Aggregate known matrix artifacts without requiring every run to finish."""
    expected: dict[Path, tuple[int, int]] = {}
    warnings: list[dict[str, str]] = []
    for config_path in sorted(root.rglob("config.json")):
        config = read_json(config_path)
        if not isinstance(config, dict):
            continue
        matrix = config.get("condition_matrix_evaluation", {}) or {}
        if not isinstance(matrix, dict) or not bool(matrix.get("enabled", False)):
            continue
        names = matrix.get("condition_names")
        target_cols = (config.get("data", {}) or {}).get("target_cols", {}) or {}
        condition_count = len(names) if isinstance(names, list) else 0
        task_count = len(target_cols) if isinstance(target_cols, dict) else 0
        expected[config_path.parent] = (condition_count, condition_count * task_count)

    manifest_paths = sorted(root.rglob("condition_matrix_evaluation/manifest.json"))
    matrix_dirs = {path.parent.parent for path in manifest_paths}
    matrix_dirs.update(expected)
    # Include interrupted evaluations that wrote a CSV but not their success manifest.
    for csv_path in root.rglob("condition_matrix_evaluation/condition_metrics.csv"):
        matrix_dirs.add(csv_path.parent.parent)
    for csv_path in root.rglob("condition_matrix_evaluation/task_metrics.csv"):
        matrix_dirs.add(csv_path.parent.parent)
    if not matrix_dirs:
        return None

    condition_frames: list[pd.DataFrame] = []
    task_frames: list[pd.DataFrame] = []
    complete_manifests = 0
    incomplete_manifests = 0
    malformed_manifests = 0
    missing_manifests = 0
    expected_condition_rows = 0
    expected_task_rows = 0
    for run_dir in sorted(matrix_dirs):
        matrix_dir = run_dir / "condition_matrix_evaluation"
        manifest_path = matrix_dir / "manifest.json"
        manifest = None
        if manifest_path.is_file():
            manifest = _matrix_manifest(manifest_path, root, warnings)
            if manifest is None:
                malformed_manifests += 1
            elif manifest.get("status") == "complete":
                complete_manifests += 1
            else:
                incomplete_manifests += 1
                warnings.append({
                    "code": "incomplete_matrix_manifest",
                    "path": str(manifest_path.relative_to(root)),
                    "message": f"status is {manifest.get('status')!r}, expected 'complete'",
                })
        else:
            missing_manifests += 1
            warnings.append({
                "code": "missing_matrix_manifest",
                "path": str(matrix_dir.relative_to(root)),
                "message": "condition-matrix success manifest is missing",
            })

        configured_counts = expected.get(run_dir)
        if configured_counts is not None:
            expected_condition_rows += configured_counts[0]
            expected_task_rows += configured_counts[1]
        elif manifest is not None:
            expected_condition_rows += int(manifest.get("expected_condition_cells", 0) or 0)
            expected_task_rows += int(manifest.get("expected_task_rows", 0) or 0)

        condition_path = matrix_dir / "condition_metrics.csv"
        if condition_path.is_file():
            condition_frames.append(
                _matrix_csv(condition_path, _CONDITION_REQUIRED, root, warnings)
            )
        else:
            warnings.append({
                "code": "missing_matrix_csv",
                "path": str(condition_path.relative_to(root)),
                "message": "condition metrics are missing",
            })
        task_path = matrix_dir / "task_metrics.csv"
        if task_path.is_file():
            task_frames.append(_matrix_csv(task_path, _TASK_REQUIRED, root, warnings))
        else:
            warnings.append({
                "code": "missing_matrix_csv",
                "path": str(task_path.relative_to(root)),
                "message": "task metrics are missing",
            })

    conditions = (
        pd.concat(condition_frames, ignore_index=True)
        if condition_frames else pd.DataFrame(columns=[*_CONDITION_REQUIRED, "source_path"])
    )
    tasks = (
        pd.concat(task_frames, ignore_index=True)
        if task_frames else pd.DataFrame(columns=[*_TASK_REQUIRED, "source_path"])
    )
    conditions, duplicate_conditions = _deduplicate_matrix_rows(
        conditions,
        ["run_name", "model", "train_condition", "test_condition"],
        warnings,
        CONDITION_MATRIX_CONDITION_OUTPUT,
    )
    tasks, duplicate_tasks = _deduplicate_matrix_rows(
        tasks,
        ["run_name", "model", "train_condition", "test_condition", "task"],
        warnings,
        CONDITION_MATRIX_TASK_OUTPUT,
    )
    conditions.to_csv(root / CONDITION_MATRIX_CONDITION_OUTPUT, index=False)
    tasks.to_csv(root / CONDITION_MATRIX_TASK_OUTPUT, index=False)
    relation_counts = {
        relation: int((conditions["evaluation_relation"] == relation).sum())
        for relation in ("matched", "rgb_stress", "cross_condition")
    } if "evaluation_relation" in conditions else {}
    complete = (
        len(conditions) == expected_condition_rows
        and len(tasks) == expected_task_rows
        and missing_manifests == 0
        and malformed_manifests == 0
        and incomplete_manifests == 0
        and duplicate_conditions == 0
        and duplicate_tasks == 0
        and not warnings
    )
    warning_counts = {
        code: sum(warning["code"] == code for warning in warnings)
        for code in sorted({warning["code"] for warning in warnings})
    }
    summary = {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "expected_runs": len(expected),
        "discovered_matrix_runs": len(matrix_dirs),
        "complete_manifests": complete_manifests,
        "incomplete_manifests": incomplete_manifests,
        "missing_manifests": missing_manifests,
        "malformed_manifests": malformed_manifests,
        "incomplete_runs": len(matrix_dirs) - complete_manifests,
        "expected_condition_rows": expected_condition_rows,
        "collected_condition_rows": len(conditions),
        "expected_task_rows": expected_task_rows,
        "collected_task_rows": len(tasks),
        "duplicate_condition_rows": duplicate_conditions,
        "duplicate_task_rows": duplicate_tasks,
        "relation_counts": relation_counts,
        "warning_counts": warning_counts,
        "warnings": warnings,
    }
    (root / CONDITION_MATRIX_SUMMARY_OUTPUT).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {len(conditions)} condition-matrix rows and {len(tasks)} task rows "
        f"to {root} ({summary['status']})"
    )
    return summary


def collect_results(root: Path) -> None:
    rows: list[dict] = []
    for summary_path in sorted(root.rglob("run_summary.json")):
        row = read_json(summary_path)
        if row is None:
            continue
        row["summary_path"] = str(summary_path.relative_to(root))
        rows.append(row)

    results = pd.DataFrame(rows)
    if not results.empty:
        sort_columns = [
            column for column in ["model", "train_feature", "train_transform", "train_strength"]
            if column in results
        ]
        if sort_columns:
            results = results.sort_values(sort_columns, na_position="first").reset_index(drop=True)
        path = root / "matched_condition_results.csv"
        results.to_csv(path, index=False)
        print(f"Wrote {len(results)} completed matched-condition runs to {path}")
        matched_long = matched_results_long(results)
        matched_long_path = root / "matched_condition_macro_f1_long.csv"
        matched_long.to_csv(matched_long_path, index=False)
        print(f"Wrote {len(matched_long)} task-level rows to {matched_long_path}")
    else:
        matched_long = pd.DataFrame()
        print("No run_summary.json files were found")

    failed: list[dict] = []
    for status_path in sorted(root.glob("*/run_status.txt")):
        status = status_path.read_text().strip()
        if status != "0":
            failed.append({"array_run": status_path.parent.name, "status": status})
    if failed:
        failed_path = root / "failed_runs.csv"
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"Recorded {len(failed)} failed runs in {failed_path}")

    cue_ratios = collect_nested_csv(
        root, "cue_suppression/macro_f1_ratios.csv",
        "rgb_model_cue_suppression_macro_f1_ratios.csv",
    )
    collect_nested_csv(
        root, "cue_suppression/test_condition_metrics.csv",
        "rgb_model_cue_suppression_test_metrics.csv",
    )
    collect_nested_csv(
        root, "cue_suppression/transform_summary.csv",
        "rgb_model_cue_suppression_transform_summary.csv",
    )
    comparison = build_comparison(matched_long, cue_ratios)
    if not comparison.empty:
        comparison_path = root / "matched_vs_rgb_stress_test.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"Wrote {len(comparison)} comparison rows to {comparison_path}")
    collect_condition_matrix(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root")
    args = parser.parse_args()
    collect_results(Path(args.results_root))
