"""Small, dependency-free data models for read-only result discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FilesystemRunState(str, Enum):
    """Status inferred solely from result files, never from the scheduler."""

    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    POSSIBLY_ACTIVE = "possibly_active"


# Compatibility name retained for the dashboard draft and downstream callers.
RunStatus = FilesystemRunState


@dataclass(frozen=True)
class WarningRecord:
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    path: str
    relative_path: str
    available: bool
    is_symlink: bool
    link_target: str | None
    size: int
    mtime_ns: int
    target_size: int | None = None
    target_mtime_ns: int | None = None


@dataclass
class MetricRecord:
    phase: str
    metric: str
    value: float
    task: str | None = None
    n: int | None = None
    source: str | None = None
    condition: str | None = None
    condition_relation: str | None = None
    canonical_key: str | None = None


@dataclass(frozen=True)
class SweepPlanEntry:
    """One row of a sweep plan with unknown launcher fields preserved."""

    run_index: int | None
    array_name: str | None
    values: dict[str, str]


@dataclass
class RunRecord:
    uid: str
    experiment_uid: str
    experiment_name: str
    array_run: str | None
    run_name: str
    path: str
    relative_path: str
    source_kind: str
    source_label: str
    source_root: str
    status: FilesystemRunState
    status_evidence: str
    updated_at: float
    model: str | None
    tasks: list[str]
    training_mode: str
    train_condition: str | None
    train_feature: str | None
    train_transform: str | None
    train_strength: float | None
    fixed_rgb_stress_evaluation: bool
    best_epoch: int | None
    best_val_score: float | None
    selection_metric: str | None
    config: dict[str, Any]
    overrides: str | None
    metrics: list[MetricRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    schema_version: str = "unknown"
    signature: str = ""
    raw_exit_status: str | None = None
    terminal_metrics_present: bool = False
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    effective_macro_f1: float | None = None
    effective_macro_f1_label: str | None = None
    epochs_ran: int | None = None
    experiment_type: str | None = None
    image_size: int | None = None
    train_condition_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    def artifact(self, kind: str) -> ArtifactRecord | None:
        return next((item for item in self.artifacts if item.kind == kind), None)

    def iter_artifacts(self, kind: str) -> list[ArtifactRecord]:
        return [item for item in self.artifacts if item.kind == kind]


@dataclass
class ExperimentRecord:
    uid: str
    name: str
    path: str
    relative_path: str
    source_kind: str
    source_label: str
    source_root: str
    updated_at: float
    expected_run_count: int | None
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    signature: str = ""
    plan_entries: list[SweepPlanEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def artifact(self, kind: str) -> ArtifactRecord | None:
        return next((item for item in self.artifacts if item.kind == kind), None)

    def iter_artifacts(self, kind: str) -> list[ArtifactRecord]:
        return [item for item in self.artifacts if item.kind == kind]


@dataclass
class DiscoveryResult:
    results_root: str
    experiments: list[ExperimentRecord]
    runs: list[RunRecord]
    warnings: list[WarningRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "results_root": self.results_root,
            "experiments": [item.to_dict() for item in self.experiments],
            "runs": [item.to_dict() for item in self.runs],
            "warnings": [asdict(item) for item in self.warnings],
        }


@dataclass
class ExperimentSnapshot:
    """Explicit single-experiment discovery result."""

    experiment: ExperimentRecord
    runs: list[RunRecord]
    warnings: list[WarningRecord] = field(default_factory=list)

    @property
    def unmaterialized_array_runs(self) -> list[str]:
        materialized = {run.array_run for run in self.runs if run.array_run}
        return [
            entry.array_name
            for entry in self.experiment.plan_entries
            if entry.array_name and entry.array_name not in materialized
        ]


ResultsSnapshot = DiscoveryResult
