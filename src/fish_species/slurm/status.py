"""Read-only filesystem and optional scheduler status reporting."""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

from ..results import WarningRecord, discover_experiment, load_json
from ..results.readers import load_csv_rows
from .submission import CommandResult


_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_ARRAY_JOB_ID = re.compile(r"^(?P<base>[1-9][0-9]*)_(?P<index>[0-9]+)$")


class SchedulerRunner(Protocol):
    def run(self, argv: Sequence[str]) -> CommandResult:
        """Run one tokenized, read-only scheduler query."""


class SubprocessSchedulerRunner:
    def run(self, argv: Sequence[str]) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            check=False,
            text=True,
            capture_output=True,
            timeout=15,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True)
class SchedulerJobRecord:
    job_id: str
    array_index: int | None
    job_name: str | None
    raw_state: str
    normalized_state: str
    exit_code: str | None
    reason: str | None
    source: str


@dataclass(frozen=True)
class RunStatusRecord:
    array_run: str | None
    run_name: str
    path: str
    filesystem_state: str
    scheduler_state: str | None
    status_evidence: str


@dataclass(frozen=True)
class StatusReport:
    results_root: str
    experiment_name: str
    expected_run_count: int | None
    materialized_run_count: int
    unmaterialized_run_count: int
    filesystem_counts: dict[str, int]
    scheduler_counts: dict[str, int]
    scheduler_available: bool
    submitted_jobs: dict[str, str]
    jobs: tuple[SchedulerJobRecord, ...]
    runs: tuple[RunStatusRecord, ...]
    warnings: tuple[WarningRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "results_root": self.results_root,
            "experiment_name": self.experiment_name,
            "expected_run_count": self.expected_run_count,
            "materialized_run_count": self.materialized_run_count,
            "unmaterialized_run_count": self.unmaterialized_run_count,
            "filesystem_counts": self.filesystem_counts,
            "scheduler_counts": self.scheduler_counts,
            "scheduler_available": self.scheduler_available,
            "submitted_jobs": self.submitted_jobs,
            "jobs": [asdict(item) for item in self.jobs],
            "runs": [asdict(item) for item in self.runs],
            "warnings": [asdict(item) for item in self.warnings],
        }


def _normalise_state(raw_state: str) -> str:
    state = raw_state.strip().upper().rstrip("+")
    if state.startswith("CANCELLED"):
        return "cancelled"
    mapping = {
        "CONFIGURING": "pending",
        "PENDING": "pending",
        "REQUEUED": "pending",
        "RESIZING": "running",
        "RUNNING": "running",
        "COMPLETING": "completing",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "BOOT_FAIL": "failed",
        "DEADLINE": "failed",
        "NODE_FAIL": "failed",
        "OUT_OF_MEMORY": "failed",
        "PREEMPTED": "failed",
        "REVOKED": "failed",
        "TIMEOUT": "timeout",
        "SUSPENDED": "unknown",
    }
    return mapping.get(state, "unknown")


def _job_identity(raw_job_id: str) -> tuple[str, int | None] | None:
    job_id = raw_job_id.strip()
    if "." in job_id:
        return None
    match = _ARRAY_JOB_ID.fullmatch(job_id)
    if match:
        return match.group("base"), int(match.group("index"))
    if _JOB_ID.fullmatch(job_id):
        return job_id, None
    return None


def _parse_rows(text: str, source: str) -> list[SchedulerJobRecord]:
    records: list[SchedulerJobRecord] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        values = [value.strip() for value in line.split("|", 4)]
        if len(values) < 4:
            continue
        identity = _job_identity(values[0])
        if identity is None:
            continue
        base, array_index = identity
        raw_state = values[2]
        exit_code = values[3] if source == "sacct" and values[3] else None
        reason_index = 4 if source == "sacct" else 3
        reason = values[reason_index] if len(values) > reason_index else None
        records.append(
            SchedulerJobRecord(
                job_id=base,
                array_index=array_index,
                job_name=values[1] or None,
                raw_state=raw_state,
                normalized_state=_normalise_state(raw_state),
                exit_code=exit_code,
                reason=reason or None,
                source=source,
            )
        )
    return records


def _load_submitted_jobs(
    root: Path, warnings: list[WarningRecord]
) -> dict[str, str]:
    receipt = root / "submission_receipt.json"
    submitted: dict[str, str] = {}
    if receipt.is_file():
        try:
            raw = load_json(receipt).get("submitted", {})
            if not isinstance(raw, dict):
                raise ValueError("submitted must be a mapping")
            for name, job_id in raw.items():
                value = str(job_id)
                if _JOB_ID.fullmatch(value):
                    submitted[str(name)] = value
                else:
                    warnings.append(
                        WarningRecord(
                            "invalid_job_id",
                            f"Ignored invalid job ID {value!r}",
                            str(receipt),
                        )
                    )
        except Exception as exc:
            warnings.append(
                WarningRecord(
                    "malformed_submission_receipt",
                    f"Could not read submission receipt: {exc}",
                    str(receipt),
                )
            )
    table = root / "submitted_jobs.tsv"
    if table.is_file():
        try:
            for row in load_csv_rows(table, max_rows=10_000):
                name = row.get("name")
                value = row.get("job_id", "")
                if not name or name in submitted:
                    continue
                if _JOB_ID.fullmatch(value):
                    submitted[name] = value
                else:
                    warnings.append(
                        WarningRecord(
                            "invalid_job_id",
                            f"Ignored invalid job ID {value!r}",
                            str(table),
                        )
                    )
        except Exception as exc:
            warnings.append(
                WarningRecord(
                    "malformed_submitted_jobs",
                    f"Could not read submitted jobs: {exc}",
                    str(table),
                )
            )
    return submitted


def _query_scheduler(
    submitted: dict[str, str],
    runner: SchedulerRunner,
    warnings: list[WarningRecord],
) -> tuple[list[SchedulerJobRecord], bool]:
    ids = sorted(set(submitted.values()), key=int)
    if not ids:
        return [], False
    joined = ",".join(ids)
    commands = (
        (
            "squeue",
            [
                "squeue",
                "--noheader",
                "--jobs",
                joined,
                "--Format=%i|%j|%T|%R",
            ],
        ),
        (
            "sacct",
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                joined,
                "--format=JobIDRaw,JobName,State,ExitCode,Reason",
            ],
        ),
    )
    by_identity: dict[tuple[str, int | None], SchedulerJobRecord] = {}
    available = False
    for source, argv in commands:
        try:
            result = runner.run(argv)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(
                WarningRecord("scheduler_unavailable", f"{source}: {exc}")
            )
            continue
        if result.returncode != 0:
            warnings.append(
                WarningRecord(
                    "scheduler_query_failed",
                    f"{source} exited {result.returncode}: {result.stderr.strip()}",
                )
            )
            continue
        available = True
        for record in _parse_rows(result.stdout, source):
            identity = (record.job_id, record.array_index)
            existing = by_identity.get(identity)
            if existing is None or source == "squeue":
                by_identity[identity] = record
    return sorted(
        by_identity.values(),
        key=lambda item: (int(item.job_id), item.array_index is None, item.array_index or -1),
    ), available


def _plan_name(entry) -> str | None:
    return entry.array_name or entry.values.get("run_name") or None


def build_status_report(
    results_root: str | Path,
    *,
    submission_root: str | Path | None = None,
    runner: SchedulerRunner | None = None,
    query_scheduler: bool = True,
    max_depth: int = 8,
) -> StatusReport:
    """Combine one discovery snapshot with optional scheduler evidence."""
    snapshot = discover_experiment(results_root, max_depth=max_depth)
    warnings = [
        *snapshot.warnings,
        *snapshot.experiment.warnings,
        *(warning for run in snapshot.runs for warning in run.warnings),
    ]
    metadata_root = Path(submission_root or results_root).expanduser().absolute()
    submitted = _load_submitted_jobs(metadata_root, warnings)
    jobs: list[SchedulerJobRecord] = []
    scheduler_available = False
    if query_scheduler and submitted:
        jobs, scheduler_available = _query_scheduler(
            submitted, runner or SubprocessSchedulerRunner(), warnings
        )

    expected_names = {
        name
        for entry in snapshot.experiment.plan_entries
        if (name := _plan_name(entry)) is not None
    }
    materialized_names = {
        run.array_run or run.run_name for run in snapshot.runs
    }
    unmaterialized = expected_names - materialized_names
    train_job_ids = {
        job_id
        for name, job_id in submitted.items()
        if name in {"train_array", "gpu_array"} or "array" in name
    }
    index_by_name = {
        name: entry.run_index
        for entry in snapshot.experiment.plan_entries
        if (name := _plan_name(entry)) is not None
    }
    scheduler_by_index = {
        record.array_index: record
        for record in jobs
        if record.job_id in train_job_ids and record.array_index is not None
    }
    run_records: list[RunStatusRecord] = []
    for run in snapshot.runs:
        array_name = run.array_run or run.run_name
        scheduler = scheduler_by_index.get(index_by_name.get(array_name))
        if (
            scheduler is not None
            and scheduler.normalized_state in {"failed", "cancelled", "timeout"}
            and run.status.value == "completed"
        ):
            warnings.append(
                WarningRecord(
                    "status_conflict",
                    "Scheduler failure conflicts with completed scientific artifacts",
                    run.path,
                )
            )
        run_records.append(
            RunStatusRecord(
                array_run=run.array_run,
                run_name=run.run_name,
                path=run.path,
                filesystem_state=run.status.value,
                scheduler_state=(scheduler.normalized_state if scheduler else None),
                status_evidence=run.status_evidence,
            )
        )
    filesystem_counts = dict(
        sorted(Counter(run.filesystem_state for run in run_records).items())
    )
    scheduler_counts = dict(
        sorted(Counter(job.normalized_state for job in jobs).items())
    )
    expected_count = snapshot.experiment.expected_run_count
    if expected_count is None and expected_names:
        expected_count = len(expected_names)
    return StatusReport(
        results_root=str(Path(results_root).expanduser().absolute()),
        experiment_name=snapshot.experiment.name,
        expected_run_count=expected_count,
        materialized_run_count=len(snapshot.runs),
        unmaterialized_run_count=len(unmaterialized),
        filesystem_counts=filesystem_counts,
        scheduler_counts=scheduler_counts,
        scheduler_available=scheduler_available,
        submitted_jobs=submitted,
        jobs=tuple(jobs),
        runs=tuple(run_records),
        warnings=tuple(warnings),
    )


__all__ = [
    "RunStatusRecord",
    "SchedulerJobRecord",
    "SchedulerRunner",
    "StatusReport",
    "SubprocessSchedulerRunner",
    "build_status_report",
]
