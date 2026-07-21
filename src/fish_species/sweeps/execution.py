"""Local and SLURM execution adapters for phased sweep runs."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..slurm.config import load_submission_config
from ..slurm.planning import RunSpec, SubmissionPlan, plan_submission
from ..slurm.rendering import write_artifact_bundle
from ..slurm.submission import (
    CommandResult,
    SbatchClient,
    SubmissionError,
    SubprocessSbatchClient,
    parse_job_id,
    submit_manifest,
)
from .core import PipelineError, pipeline_paths, resolve_pipeline_path


def _yaml_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(dict(config), sort_keys=True).encode()).hexdigest()


def _submission_config(
    config_path: str, cluster_config: str | None, phase_output: Path,
    max_active: int,
) -> dict[str, Any]:
    overrides = [
        f"slurm.paths.results_root={phase_output}",
        f"slurm.array.max_active={max_active}",
        "slurm.collection.enabled=false",
    ]
    config = load_submission_config(config_path, cluster_config, overrides)
    config.setdefault("slurm", {}).setdefault("collection", {})["enabled"] = False
    return config


def build_phase_submission_plan(
    records: Sequence[Mapping[str, Any]],
    cluster_config: str | None,
    phase_output: Path,
    max_active: int,
) -> tuple[SubmissionPlan, dict[str, Any]]:
    """Adapt pipeline records to the repository's immutable SLURM plan."""
    if not records:
        raise PipelineError("cannot submit an empty phase")
    config = _submission_config(
        str(records[0]["resolved_config_path"]), cluster_config,
        phase_output, max_active,
    )
    base = plan_submission(config)
    specs = []
    for index, record in enumerate(records):
        resolved = load_submission_config(
            str(record["resolved_config_path"]), cluster_config,
            ["slurm.collection.enabled=false"],
        )
        resolved.pop("slurm", None)
        specs.append(RunSpec(
            index=index, run_id=str(record["run_id"]),
            model=str((resolved.get("model", {}) or {}).get("name", "model")),
            training_condition=str(record.get("condition", "pipeline")),
            training_transform=str((resolved.get("input_condition", {}) or {}).get("transform", "original")),
            experiment_type="pipeline", training_mode=str((resolved.get("training", {}) or {}).get("mode", "multitask")),
            overrides=(), trainer_overrides=("sweep.enabled=false", "matched_condition_training.enabled=false"),
            output_relpath=str(record["run_id"]), resolved_config=resolved,
            config_sha256=_yaml_hash(resolved),
        ))
    plan = SubmissionPlan(
        schema_version=base.schema_version, experiment_type="pipeline",
        cluster_profile=base.cluster_profile, results_root=str(phase_output),
        array_size=len(specs), array_max_active=max_active,
        models=tuple(dict.fromkeys(item.model for item in specs)),
        conditions=tuple(dict.fromkeys(item.training_condition for item in specs)),
        training_modes=tuple(dict.fromkeys(item.training_mode for item in specs)),
        run_specs=tuple(specs), dependencies=base.dependencies,
        canonical_trainer_command=base.canonical_trainer_command,
        resolved_config_sha256=_yaml_hash(config),
    )
    return plan, config


def render_phase_bundle(
    document: Mapping[str, Any], phase_name: str,
    records: Sequence[Mapping[str, Any]], cluster_config: str | None,
    attempt: int,
) -> tuple[Path, list[list[str]]]:
    """Render one immutable attempt bundle and return its manifest path."""
    root, _ = pipeline_paths(document)
    phase_output = root / "phases" / phase_name / "runs"
    max_active = int(document["pipeline"].get("execution", {}).get("array_max_active", 1))
    plan, config = build_phase_submission_plan(records, cluster_config, phase_output, max_active)
    artifact_dir = root / "phases" / phase_name / "slurm" / f"attempt_{attempt:03d}"
    manifest = write_artifact_bundle(plan, config, artifact_dir)
    from ..slurm.submission import build_submission_commands
    return artifact_dir / "submission_manifest.json", build_submission_commands(manifest)


def submit_phase_slurm(
    document: Mapping[str, Any], phase_name: str,
    records: Sequence[Mapping[str, Any]], cluster_config: str | None,
    attempt: int, *, client: SbatchClient | None = None,
) -> tuple[dict[str, str], Path]:
    """Render and submit one phase array through the canonical scheduler API."""
    manifest_path, _ = render_phase_bundle(document, phase_name, records, cluster_config, attempt)
    return submit_manifest(manifest_path, client=client), manifest_path


def submit_auto_advance(
    document: Mapping[str, Any], train_job_id: str, cluster_config: str | None,
    *, client: SbatchClient | None = None,
) -> str:
    """Submit the lightweight afterany collector/advancer job."""
    scheduler = client or SubprocessSbatchClient()
    pipeline_file = str(Path(str(document["_pipeline_file"])).resolve())
    resolved = load_submission_config(
        resolve_pipeline_path(document, str(document["pipeline"]["base_config"])),
        cluster_config,
        ["slurm.collection.enabled=false"],
    )
    slurm = resolved.get("slurm", {}) or {}
    project_root = Path(str((slurm.get("paths", {}) or {}).get(
        "project_root", Path(__file__).resolve().parents[3]
    )))
    child = [
        "python", "-m", "fish_species.sweeps.pipeline", "advance",
        "--pipeline", pipeline_file, "--submit",
    ]
    if cluster_config:
        child.extend(["--cluster-config", str(Path(cluster_config).resolve())])
    environment = slurm.get("environment", {}) or {}
    conda_sh = environment.get("conda_sh")
    conda_env = environment.get("conda_env")
    activation = ""
    if conda_sh and conda_env:
        activation = (
            f"source {shlex.quote(str(conda_sh))} && "
            f"conda activate {shlex.quote(str(conda_env))} && "
        )
    shell_command = (
        activation
        + f"export PYTHONPATH={shlex.quote(str(project_root / 'src'))}; "
        + shlex.join(child)
    )
    command = ["bash", "-lc", shell_command]
    execution = document["pipeline"].get("execution", {}) or {}
    argv = [
        "sbatch", "--parsable", f"--dependency=afterany:{train_job_id}",
        "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
        f"--mem={int(execution.get('collector_memory_mib', 2048))}",
        f"--time={execution.get('collector_time_limit', '00:15:00')}",
        f"--job-name={document['pipeline']['name']}-advance",
        f"--chdir={project_root}",
        "--wrap=" + shlex.join(command),
    ]
    result = scheduler.run(argv)
    if result.returncode:
        raise SubmissionError(f"collector submission failed: {result.stderr.strip()}")
    return parse_job_id(result.stdout)


def execute_local_records(
    records: Sequence[Mapping[str, Any]], *, environment: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Run unresolved records sequentially through the canonical trainer."""
    env = dict(os.environ if environment is None else environment)
    source_root = str((Path(__file__).resolve().parents[2]))
    env["PYTHONPATH"] = source_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    results: dict[str, int] = {}
    for record in records:
        output = Path(str(record["expected_output_directory"]))
        command = [
            sys.executable, "-m", "fish_species.training", "--config",
            str(record["resolved_config_path"]), "--single-run",
        ]
        completed = subprocess.run(command, env=env, check=False)
        results[str(record["run_id"])] = completed.returncode
        status_path = output / "run_status.json"
        if completed.returncode and not status_path.exists():
            output.mkdir(parents=True, exist_ok=True)
            status_path.write_text(json.dumps({
                "status": "failed", "exit_code": completed.returncode,
                "configuration_hash": record["configuration_hash"],
                "failure_category": "training_failure",
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def cancel_jobs(job_ids: Sequence[str], *, dry_run: bool = False) -> list[list[str]]:
    """Cancel explicit pipeline-owned scheduler job IDs without deleting state."""
    commands = [["scancel", str(job_id)] for job_id in dict.fromkeys(job_ids) if str(job_id).isdigit()]
    if not dry_run:
        for command in commands:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode:
                raise PipelineError(f"scancel failed for {command[-1]}: {completed.stderr.strip()}")
    return commands


def active_slurm_jobs(job_ids: Sequence[str]) -> list[str]:
    """Return recorded job IDs still visible to SLURM's active-job query."""
    identifiers = [str(item) for item in dict.fromkeys(job_ids) if str(item).isdigit()]
    if not identifiers:
        return []
    try:
        completed = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(identifiers), "--format=%A"],
            check=False, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise PipelineError(
            "squeue is unavailable; run this command on a SLURM login node"
        ) from exc
    if completed.returncode:
        raise PipelineError(
            "could not verify whether the training array is still active; "
            "retry on a SLURM login node or use --force after checking it manually: "
            + completed.stderr.strip()
        )
    active = []
    for line in completed.stdout.splitlines():
        value = line.strip().split("_", 1)[0]
        if value in identifiers and value not in active:
            active.append(value)
    return active


__all__ = [
    "build_phase_submission_plan", "cancel_jobs", "execute_local_records",
    "render_phase_bundle", "submit_auto_advance", "submit_phase_slurm",
    "active_slurm_jobs",
]
