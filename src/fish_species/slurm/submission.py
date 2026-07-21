"""Deterministic argv construction and injectable SLURM submission."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from typing import Protocol
from typing import Sequence

from .rendering import RenderError
from .rendering import verify_artifact_bundle


_JOB_ID = re.compile(r"^(?P<id>[1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?$")


class SubmissionError(RuntimeError):
    """A rendered job graph could not be submitted exactly as planned."""

    def __init__(self, message: str, *, submitted: Mapping[str, str] | None = None):
        super().__init__(message)
        self.submitted = dict(submitted or {})


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str = ""


class SbatchClient(Protocol):
    def run(self, argv: Sequence[str]) -> CommandResult:
        """Execute one tokenised sbatch command."""


class SubprocessSbatchClient:
    """Production scheduler client; never invokes a command shell."""

    def run(self, argv: Sequence[str]) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            check=False,
            text=True,
            capture_output=True,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


class RecordingSbatchClient:
    """Small deterministic scheduler double for tests and local demonstrations."""

    def __init__(self, job_ids: Sequence[str] | None = None):
        self.job_ids = list(job_ids or [])
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str]) -> CommandResult:
        self.calls.append(list(argv))
        index = len(self.calls) - 1
        job_id = self.job_ids[index] if index < len(self.job_ids) else str(1000 + index)
        return CommandResult(returncode=0, stdout=f"{job_id}\n")


def parse_job_id(stdout: str) -> str:
    """Parse strict `sbatch --parsable` output."""
    value = stdout.strip()
    match = _JOB_ID.fullmatch(value)
    if match is None:
        raise SubmissionError(f"Invalid sbatch --parsable response: {value!r}")
    return match.group("id")


def _dependency_argument(
    dependencies: Sequence[Mapping[str, str]],
    job_ids: Mapping[str, str],
    *,
    symbolic: bool,
) -> str | None:
    grouped: dict[str, list[str]] = {}
    for dependency in dependencies:
        name = dependency["job"]
        kind = dependency["kind"]
        if kind not in {"afterok", "afterany"}:
            raise SubmissionError(f"Unsupported dependency kind: {kind!r}")
        if symbolic:
            job_id = f"@{name}"
        else:
            try:
                job_id = job_ids[name]
            except KeyError as exc:
                raise SubmissionError(
                    f"Dependency {name!r} has not been submitted"
                ) from exc
        grouped.setdefault(kind, []).append(job_id)
    if not grouped:
        return None
    return ",".join(
        f"{kind}:{':'.join(ids)}" for kind, ids in grouped.items()
    )


def build_sbatch_argv(
    job: Mapping[str, object],
    artifact_root: str | Path,
    job_ids: Mapping[str, str] | None = None,
    *,
    symbolic: bool = False,
) -> list[str]:
    """Build one exact scheduler argv without word splitting or a shell."""
    root = Path(artifact_root).resolve()
    script = (root / str(job["script"])).resolve()
    if root not in script.parents or not script.is_file() or script.is_symlink():
        raise SubmissionError(f"Unsafe or missing rendered script: {script}")

    argv = [
        "sbatch",
        "--parsable",
        f"--account={job['account']}",
        f"--nodes={job['nodes']}",
        f"--ntasks={job['ntasks']}",
        f"--cpus-per-task={job['cpus_per_task']}",
        f"--mem={job['memory_mib']}",
        f"--time={job['time_limit']}",
        f"--job-name={job['job_name']}",
        f"--output={job['stdout']}",
        f"--error={job['stderr']}",
    ]
    partition = job.get("partition")
    if partition:
        argv.insert(3, f"--partition={partition}")
    excluded = job.get("exclude_nodes", [])
    if not isinstance(excluded, list) or any(
        not isinstance(node, str) or not node for node in excluded
    ):
        raise SubmissionError(f"Job {job['name']} exclude_nodes must be a string list")
    if excluded:
        argv.append("--exclude=" + ",".join(excluded))
    gpus = int(job.get("gpus_per_task", 0))
    if gpus:
        argv.append(f"--gres=gpu:{gpus}")
    if job.get("nodelist"):
        argv.append(f"--nodelist={job['nodelist']}")
    if job.get("array"):
        argv.append(f"--array={job['array']}")

    dependency = _dependency_argument(
        job.get("dependencies", []),
        job_ids or {},
        symbolic=symbolic,
    )
    if dependency:
        argv.append(f"--dependency={dependency}")

    exports = job.get("exports", {})
    if not isinstance(exports, dict):
        raise SubmissionError(f"Job {job['name']} exports must be a mapping")
    export_values = []
    for key, value in exports.items():
        if value is None:
            export_values.append(str(key))
        else:
            export_values.append(f"{key}={value}")
    if export_values:
        argv.append("--export=" + ",".join(export_values))

    extra_args = job.get("extra_args", [])
    if not isinstance(extra_args, list) or any(
        not isinstance(item, str) for item in extra_args
    ):
        raise SubmissionError(f"Job {job['name']} extra_args must be a string list")
    argv.extend(extra_args)
    argv.append(str(script))
    return argv


def build_submission_commands(manifest: Mapping[str, object]) -> list[list[str]]:
    """Return the dry-run command list with symbolic dependency references."""
    root = manifest.get("artifact_root")
    jobs = manifest.get("jobs")
    if not isinstance(root, str) or not isinstance(jobs, list):
        raise SubmissionError("Submission manifest is missing artifact_root/jobs")
    return [build_sbatch_argv(job, root, symbolic=True) for job in jobs]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_receipt(
    root: Path,
    submitted: Mapping[str, str],
    calls: Sequence[Mapping[str, object]],
    error: str | None = None,
) -> None:
    receipt = {
        "schema_version": 1,
        "submitted": dict(submitted),
        "calls": list(calls),
        "error": error,
    }
    _atomic_json(root / "submission_receipt.json", receipt)
    temporary = root / f".submitted_jobs.tsv.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["name", "job_id"])
        writer.writerows(submitted.items())
    os.replace(temporary, root / "submitted_jobs.tsv")


def submit_manifest(
    manifest_path: str | Path,
    *,
    client: SbatchClient | None = None,
) -> dict[str, str]:
    """Verify and submit a rendered graph, retaining partial-failure evidence."""
    try:
        manifest = verify_artifact_bundle(manifest_path)
    except RenderError as exc:
        raise SubmissionError(str(exc)) from exc
    root = Path(manifest_path).resolve().parent
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise SubmissionError("Submission manifest has no jobs")

    scheduler = client or SubprocessSbatchClient()
    submitted: dict[str, str] = {}
    calls: list[dict[str, object]] = []
    for job in jobs:
        name = str(job["name"])
        argv = build_sbatch_argv(job, root, submitted)
        result = scheduler.run(argv)
        call = {
            "job": name,
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        calls.append(call)
        if result.returncode != 0:
            message = (
                f"sbatch failed for {name!r} with exit {result.returncode}: "
                f"{result.stderr.strip()}"
            )
            _write_receipt(root, submitted, calls, message)
            raise SubmissionError(message, submitted=submitted)
        try:
            submitted[name] = parse_job_id(result.stdout)
        except SubmissionError as exc:
            _write_receipt(root, submitted, calls, str(exc))
            raise SubmissionError(str(exc), submitted=submitted) from exc
        _write_receipt(root, submitted, calls)
    return submitted


__all__ = [
    "CommandResult",
    "RecordingSbatchClient",
    "SbatchClient",
    "SubmissionError",
    "SubprocessSbatchClient",
    "build_sbatch_argv",
    "build_submission_commands",
    "parse_job_id",
    "submit_manifest",
]
