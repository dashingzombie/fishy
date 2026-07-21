"""Exact adapter for existing, schema-stable result aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..experiments.result_collection import collect_results as collect_dual_results


DUAL_OUTPUT_NAMES = (
    "matched_condition_results.csv",
    "matched_condition_macro_f1_long.csv",
    "failed_runs.csv",
    "rgb_model_cue_suppression_macro_f1_ratios.csv",
    "rgb_model_cue_suppression_test_metrics.csv",
    "rgb_model_cue_suppression_transform_summary.csv",
    "matched_vs_rgb_stress_test.csv",
    "condition_matrix_evaluations.csv",
    "condition_matrix_task_metrics.csv",
    "condition_matrix_collection_summary.json",
)
STANDARD_OUTPUT_NAMES = ("multi_run_results.csv",)


class CollectionError(ValueError):
    """A requested aggregation has no safe schema-preserving adapter."""


@dataclass(frozen=True)
class CollectionReport:
    results_root: str
    kind: str
    output_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "results_root": self.results_root,
            "kind": self.kind,
            "output_paths": list(self.output_paths),
        }


def _normalise_kind(kind: str) -> str:
    value = kind.strip().lower().replace("_", "-")
    aliases = {
        "dual-cue": "dual-cue",
        "matched-condition": "dual-cue",
        "rgb-stress": "dual-cue",
        "matched-and-rgb-stress": "dual-cue",
        "standard": "standard",
    }
    if value in aliases:
        return aliases[value]
    raise CollectionError(f"Unknown collection kind: {kind!r}")


def _detect_kind(root: Path) -> str:
    manifest_path = root / "condition_manifest.json"
    if manifest_path.is_file():
        try:
            experiment_type = str(
                json.loads(manifest_path.read_text()).get("experiment_type", "")
            ).strip().lower().replace("_", "-")
        except Exception:
            experiment_type = ""
        if experiment_type in {
            "dual-cue",
            "matched-condition",
            "matched-and-rgb-stress",
        }:
            return "dual-cue"
        if experiment_type in {"standard", "hierarchy", "persistent-hierarchy"}:
            return "standard"
    dual_markers = (
        "dual_cue_experiment_plan.json",
        "matched_condition_results.csv",
        "matched_vs_rgb_stress_test.csv",
    )
    if any((root / marker).is_file() for marker in dual_markers):
        return "dual-cue"
    if any(root.rglob("multi_run_results.csv")):
        return "standard"
    raise CollectionError(
        "Could not safely identify a result root; pass kind='dual-cue' or "
        "kind='standard' explicitly"
    )


def collect_standard_results(root: Path) -> None:
    """Preserve the former inline collector, including its rerun semantics."""
    files = list(root.rglob("multi_run_results.csv"))
    frames = [pd.read_csv(path) for path in files]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            root / "multi_run_results.csv", index=False
        )


def collect_existing_results(
    results_root: str | Path,
    *,
    kind: str = "auto",
) -> CollectionReport:
    """Run the exact canonical adapter selected for an existing result root."""
    root = Path(results_root).expanduser().absolute()
    if not root.is_dir():
        raise CollectionError(f"Results root is not a directory: {root}")
    selected_kind = _detect_kind(root) if kind == "auto" else _normalise_kind(kind)
    if selected_kind == "dual-cue":
        collect_dual_results(root)
        names = DUAL_OUTPUT_NAMES
    elif selected_kind == "standard":
        collect_standard_results(root)
        names = STANDARD_OUTPUT_NAMES
    else:
        raise CollectionError(f"Unsupported collection kind: {selected_kind}")
    outputs = tuple(
        str(root / name) for name in names if (root / name).is_file()
    )
    return CollectionReport(
        results_root=str(root),
        kind=selected_kind,
        output_paths=outputs,
    )


__all__ = [
    "CollectionError",
    "CollectionReport",
    "DUAL_OUTPUT_NAMES",
    "STANDARD_OUTPUT_NAMES",
    "collect_standard_results",
    "collect_existing_results",
]
