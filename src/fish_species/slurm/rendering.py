"""Strict rendering and atomic artifact writing for canonical SLURM plans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import SlurmConfigError
from .config import parse_memory
from .planning import SubmissionPlan


_TOKEN = re.compile(r"@@([A-Z][A-Z0-9_]*)@@")
_TEMPLATE_ROOT = Path(__file__).resolve().parents[3] / "slurm" / "templates"
_TEMPLATES = frozenset(
    {
        "cache_build_job.sh.tmpl",
        "persistent_cache_array_job.sh.tmpl",
        "node_local_setup_job.sh.tmpl",
        "node_local_array_job.sh.tmpl",
        "job_local_cue_array_job.sh.tmpl",
        "node_local_cleanup_job.sh.tmpl",
        "result_collector_job.sh.tmpl",
    }
)


class RenderError(SlurmConfigError):
    """A plan could not be rendered without ambiguity or unsafe substitution."""


def shell_quote(value: object) -> str:
    """Return one shell-safe scalar token."""
    if isinstance(value, (dict, list, tuple, set)):
        raise RenderError("Structured values cannot be inserted into shell templates")
    return shlex.quote(str(value))


def shell_join(argv: tuple[str, ...] | list[str]) -> str:
    """Return a shell-safe command from an already-tokenised argv."""
    if not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise RenderError("Commands must be non-empty sequences of safe strings")
    return shlex.join(argv)


def render_template(template_name: str, context: Mapping[str, str]) -> str:
    """Render a registered template and reject incomplete or surplus context."""
    if template_name not in _TEMPLATES:
        raise RenderError(f"Unknown SLURM template: {template_name!r}")
    template_path = _TEMPLATE_ROOT / template_name
    try:
        source = template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RenderError(f"SLURM template is missing: {template_path}") from exc

    placeholders = set(_TOKEN.findall(source))
    supplied = set(context)
    missing = placeholders - supplied
    unknown = supplied - placeholders
    if missing:
        raise RenderError(
            f"Missing {template_name} placeholder(s): {', '.join(sorted(missing))}"
        )
    if unknown:
        raise RenderError(
            f"Unknown {template_name} context key(s): {', '.join(sorted(unknown))}"
        )

    def substitute(match: re.Match[str]) -> str:
        value = context[match.group(1)]
        if not isinstance(value, str) or "\x00" in value:
            raise RenderError(
                f"Template value {match.group(1)} must be a NUL-free string"
            )
        return value

    rendered = _TOKEN.sub(substitute, source)
    unresolved = _TOKEN.findall(rendered)
    if unresolved:
        raise RenderError(
            f"Unresolved {template_name} placeholder(s): "
            + ", ".join(sorted(set(unresolved)))
        )
    return rendered


def _write(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_copy_mode(value: object) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return str(value)


def _select_array_template(config: dict[str, Any]) -> str:
    slurm = config["slurm"]
    mode = str(slurm.get("scratch", {}).get("mode", "none"))
    expansion = str(slurm.get("planning", {}).get("external_expansion", "sweep"))
    if mode == "node_local":
        return "node_local_array_job.sh.tmpl"
    if expansion == "dual_cue":
        return "job_local_cue_array_job.sh.tmpl"
    return "persistent_cache_array_job.sh.tmpl"


def _array_output_name(template_name: str, config: dict[str, Any]) -> str:
    """Preserve generated filenames while sharing one node-local body."""
    if template_name != "node_local_array_job.sh.tmpl":
        return template_name.removesuffix(".tmpl")
    expansion = str(
        config.get("slurm", {}).get("planning", {}).get(
            "external_expansion", "sweep"
        )
    )
    if expansion == "dual_cue":
        return "node_local_cue_array_job.sh"
    return "node_local_training_array_job.sh"


def _collector_kind(config: dict[str, Any]) -> str:
    slurm = config.get("slurm", {}) or {}
    configured = (slurm.get("collection", {}) or {}).get("kind")
    if configured not in {None, "auto"}:
        return str(configured)
    expansion = str((slurm.get("planning", {}) or {}).get("external_expansion", "sweep"))
    if expansion == "dual_cue":
        return "dual-cue"
    return "standard"


def _array_context(
    plan: SubmissionPlan,
    config: dict[str, Any],
    artifact_root: Path,
    *,
    node_local: bool,
) -> dict[str, str]:
    slurm = config["slurm"]
    paths = slurm.get("paths", {})
    scratch = slurm.get("scratch", {})
    environment = slurm.get("environment", {})
    configured_project = Path(str(paths.get("project_root", "."))).resolve()
    configured_data = str(paths.get("data_root", "data"))
    scratch_root = str(scratch.get("root", "/tmp/fish_species"))
    if node_local:
        runtime_project = f"{scratch_root.rstrip('/')}/project"
        runtime_data = f"{scratch_root.rstrip('/')}/data"
        runtime_output = f"{scratch_root.rstrip('/')}/outputs"
        runtime_cache = f"{scratch_root.rstrip('/')}/image_cache"
        runtime_metadata = f"{runtime_data}/label_train.json"
    else:
        runtime_project = str(configured_project)
        runtime_data = configured_data
        runtime_output = f"{scratch_root.rstrip('/')}/outputs"
        runtime_cache = str(paths.get("cache_root", "cache/images"))
        runtime_metadata = str(paths.get("metadata_csv", "metadata.csv"))

    command = list(plan.canonical_trainer_command)
    if "--config" in command:
        index = command.index("--config")
        del command[index : index + 2]
    command = [item for item in command if item != "--single-run"]
    gpu_count = int(slurm.get("gpus_per_task", 1))
    distributed = bool(
        (config.get("training", {}).get("distributed", {}) or {}).get(
            "enabled", False
        )
    )
    if gpu_count > 1:
        if not distributed:
            raise RenderError(
                "slurm.gpus_per_task>1 requires training.distributed.enabled=true"
            )
        command = [
            "torchrun",
            "--standalone",
            "--nnodes=1",
            f"--nproc_per_node={gpu_count}",
            "-m",
            "fish_species.training",
        ]
    return {
        "ARTIFACT_ROOT": shell_quote(artifact_root),
        "CACHE_ROOT": shell_quote(runtime_cache),
        "CACHE_READY_MARKER": shell_quote(
            f"{scratch_root.rstrip('/')}/IMAGE_CACHE_READY"
            if node_local
            else f"{runtime_cache.rstrip('/')}/CACHE_READY"
        ),
        "CONDA_ENV": shell_quote(environment.get("conda_env", "fishspecies")),
        "CONDA_SH": shell_quote(environment.get("conda_sh", "")),
        "COPY_CACHE_TO_TMP": shell_quote(
            _normalise_copy_mode(scratch.get("copy_cache_to_tmp", 0))
        ),
        "DATA_ROOT": shell_quote(runtime_data),
        "METADATA_CSV": shell_quote(runtime_metadata),
        "PROJECT_ROOT": shell_quote(runtime_project),
        "RESOLVED_CONFIG_DIR": shell_quote(artifact_root / "resolved_configs"),
        "RESULTS_ROOT": shell_quote(Path(plan.results_root).resolve()),
        "RUN_INDEX_FILE": shell_quote(artifact_root / "run_index.tsv"),
        "RUN_OUTPUT_ROOT": shell_quote(runtime_output),
        "RUN_SPECS_DIR": shell_quote(artifact_root / "run_specs"),
        "SCRATCH_ROOT": shell_quote(scratch_root),
        "TMP_RESERVE_GB": shell_quote(scratch.get("tmp_reserve_gb", 0)),
        "TRAIN_COMMAND": shell_join(command),
    }


def _resource_job(
    *,
    name: str,
    role: str,
    script: str,
    slurm: dict[str, Any],
    stage: Mapping[str, Any] | None = None,
    node: str | None = None,
    array: str | None = None,
    dependencies: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    resource = dict(slurm)
    if stage is not None:
        resource.update(stage)
    if stage is not None and "partition" in stage:
        configured_partition = stage["partition"]
        partition = "" if configured_partition is None else str(configured_partition)
    else:
        partition = str(slurm.get("partition", ""))
    logs = Path(str(slurm.get("logging", {}).get("directory", "logs/slurm")))
    safe_name = name.replace(":", "_")
    return {
        "name": name,
        "role": role,
        "script": script,
        "account": str(slurm.get("account", "")),
        "partition": partition,
        "nodes": int(resource.get("nodes", slurm.get("nodes", 1))),
        "ntasks": int(resource.get("ntasks", slurm.get("ntasks", 1))),
        "cpus_per_task": int(resource.get("cpus_per_task", 1)),
        "memory_mib": parse_memory(resource.get("memory", 1024)),
        "time_limit": str(resource.get("time_limit", "01:00:00")),
        "gpus_per_task": (
            int(slurm.get("gpus_per_task", 1)) if role == "train_array" else 0
        ),
        "nodelist": node,
        "array": array,
        "dependencies": dependencies or [],
        "job_name": f"fish_{safe_name}",
        "stdout": str(logs / f"{safe_name}_%A_%a.out" if array else logs / f"{safe_name}_%j.out"),
        "stderr": str(logs / f"{safe_name}_%A_%a.err" if array else logs / f"{safe_name}_%j.err"),
        "exports": {"ALL": None},
        "extra_args": list(slurm.get("submission", {}).get("extra_sbatch_args", [])) if role == "train_array" else [],
        "exclude_nodes": (
            list(slurm.get("submission", {}).get("exclude_nodes", []))
            if role == "train_array"
            else []
        ),
    }


def _build_jobs(
    plan: SubmissionPlan,
    config: dict[str, Any],
    script_names: dict[str, str],
) -> list[dict[str, Any]]:
    slurm = config["slurm"]
    scratch = slurm.get("scratch", {})
    nodes = list(scratch.get("nodes", []))
    jobs: list[dict[str, Any]] = []
    setup_names = []
    if bool(slurm.get("setup", {}).get("enabled", False)):
        for node in nodes:
            name = f"setup:{node}"
            setup_names.append(name)
            jobs.append(
                _resource_job(
                    name=name,
                    role="setup",
                    script=script_names["setup"],
                    slurm=slurm,
                    stage=slurm.get("setup", {}),
                    node=node,
                )
            )

    array_dependencies = [
        {"job": name, "kind": "afterok"} for name in setup_names
    ]
    jobs.append(
        _resource_job(
            name="train_array",
            role="train_array",
            script=script_names["array"],
            slurm=slurm,
            array=f"0-{plan.array_size - 1}%{plan.array_max_active}",
            dependencies=array_dependencies,
        )
    )
    wandb_config = config.get("wandb", {}) or {}
    if bool(wandb_config.get("enabled", False)):
        train_job = jobs[-1]
        train_job["exports"].update(
            {
                "WANDB_ENABLED": "true",
                "WANDB_PROJECT": str(wandb_config.get("project") or "fish-species"),
                "WANDB_ENTITY": str(wandb_config.get("entity") or ""),
                "WANDB_MODE": str(wandb_config.get("mode") or "online"),
                "WANDB_RUN_GROUP": str(
                    wandb_config.get("group") or Path(plan.results_root).name
                ),
            }
        )

    if bool(slurm.get("collection", {}).get("enabled", False)):
        jobs.append(
            _resource_job(
                name="collect",
                role="collect",
                script=script_names["collect"],
                slurm=slurm,
                stage=slurm.get("collection", {}),
                dependencies=[{"job": "train_array", "kind": "afterany"}],
            )
        )

    if bool(slurm.get("cleanup", {}).get("enabled", False)):
        cleanup_nodes = nodes if slurm.get("cleanup", {}).get("per_node") else [None]
        for node in cleanup_nodes:
            name = f"cleanup:{node}" if node else "cleanup"
            jobs.append(
                _resource_job(
                    name=name,
                    role="cleanup",
                    script=script_names["cleanup"],
                    slurm=slurm,
                    stage=slurm.get("cleanup", {}),
                    node=node,
                    dependencies=[{"job": "train_array", "kind": "afterany"}],
                )
            )
    return jobs


def _validate_job_graph(plan: SubmissionPlan, jobs: list[dict[str, Any]]) -> None:
    """Ensure rendered concrete jobs preserve the core plan's abstract DAG."""
    role_by_name = {job["name"]: job["role"] for job in jobs}
    concrete = set()
    for job in jobs:
        for dependency in job["dependencies"]:
            upstream_role = role_by_name[dependency["job"]]
            concrete.add((upstream_role, job["role"], dependency["kind"]))
    expected = {
        (dependency.upstream, dependency.downstream, dependency.kind)
        for dependency in plan.dependencies
    }
    if concrete != expected:
        raise RenderError(
            "Rendered submission DAG differs from the validated core plan: "
            f"expected={sorted(expected)!r}, rendered={sorted(concrete)!r}"
        )


def _git_metadata() -> dict[str, object]:
    """Read repository provenance, degrading explicitly outside a Git worktree."""
    repository = Path(__file__).resolve().parents[3]
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            text=True,
            capture_output=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return {
            "commit": "unknown",
            "dirty": None,
            "warning": f"Git provenance unavailable: {exc}",
        }
    if commit_result.returncode != 0 or status_result.returncode != 0:
        detail = (commit_result.stderr or status_result.stderr).strip()
        return {
            "commit": "unknown",
            "dirty": None,
            "warning": f"Git provenance unavailable: {detail or 'unknown error'}",
        }
    dirty = bool(status_result.stdout.strip())
    return {
        "commit": commit_result.stdout.strip(),
        "dirty": dirty,
        "warning": (
            "Repository has uncommitted changes; rendered artifacts may not "
            "correspond exactly to the recorded commit."
            if dirty
            else None
        ),
    }


def _execution_metadata(
    plan: SubmissionPlan,
    config: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, object]:
    slurm = config["slurm"]
    return {
        "git": _git_metadata(),
        "experiment_type": plan.experiment_type,
        "cluster_profile": plan.cluster_profile,
        "training_modes": list(plan.training_modes),
        "trainer_selection": "configuration",
        "config_hashes": {
            "submission": plan.resolved_config_sha256,
            "runs": {
                spec.run_id: spec.config_sha256 for spec in plan.run_specs
            },
        },
        "counts": {
            "models": len(plan.models),
            "conditions": len(plan.conditions),
            "runs": plan.array_size,
        },
        "models": list(plan.models),
        "conditions": list(plan.conditions),
        "resources": {
            "account": slurm.get("account"),
            "partition": slurm.get("partition"),
            "nodes": slurm.get("nodes"),
            "ntasks": slurm.get("ntasks"),
            "cpus_per_task": slurm.get("cpus_per_task"),
            "memory_mib": parse_memory(slurm.get("memory", 1024)),
            "time_limit": slurm.get("time_limit"),
            "gpus_per_task": slurm.get("gpus_per_task"),
            "array_max_active": plan.array_max_active,
        },
        "paths": dict(slurm.get("paths", {})),
        "canonical_trainer_command": list(plan.canonical_trainer_command),
        "dependency_graph": [
            {
                "job": job["name"],
                "role": job["role"],
                "dependencies": job["dependencies"],
            }
            for job in jobs
        ],
        "collector": {
            "schema_version": 1,
            "enabled": bool(slurm.get("collection", {}).get("enabled", False)),
            "kind": _collector_kind(config),
        },
    }


def _render_bundle(
    plan: SubmissionPlan,
    config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    _write(root / "launch_plan.json", _json(plan.as_dict()))
    resolved_config_text = yaml.safe_dump(config, sort_keys=True)
    _write(root / "resolved_submission_config.yaml", resolved_config_text)
    _write(root / "resolved_config.yaml", resolved_config_text)
    _write(
        root / "condition_manifest.json",
        _json(
            {
                "experiment_type": plan.experiment_type,
                "training_modes": list(plan.training_modes),
                "trainer_selection": "configuration",
                "conditions": list(plan.conditions),
                "models": list(plan.models),
                "expected_run_count": plan.array_size,
            }
        ),
    )
    slurm = config["slurm"]
    launcher_settings = {
        "CLUSTER_PROFILE": plan.cluster_profile,
        "RESULTS_ROOT": str(Path(plan.results_root).resolve()),
        "PROJECT_ROOT": slurm.get("paths", {}).get("project_root", "."),
        "DATA_ROOT": slurm.get("paths", {}).get("data_root", "data"),
        "CACHE_ROOT": slurm.get("paths", {}).get("cache_root", "cache/images"),
        "ACCOUNT": slurm.get("account", ""),
        "PARTITION": slurm.get("partition", ""),
        "MAX_ACTIVE": plan.array_max_active,
        "CONDA_ENV": slurm.get("environment", {}).get("conda_env", ""),
    }
    _write(
        root / "launcher_settings.txt",
        "".join(f"{key}={value}\n" for key, value in launcher_settings.items()),
    )

    plan_lines = ["run_index\trun_name\toverrides"]
    index_lines = ["run_index\trun_id"]
    for spec in plan.run_specs:
        _write(root / "run_specs" / f"{spec.run_id}.args", spec.args_text)
        _write(
            root / "resolved_configs" / f"{spec.run_id}.yaml",
            yaml.safe_dump(spec.resolved_config, sort_keys=True),
        )
        plan_lines.append(
            f"{spec.index}\t{spec.run_id}\t{' '.join(spec.overrides) or '<no overrides>'}"
        )
        index_lines.append(f"{spec.index}\t{spec.run_id}")
    _write(root / "sweep_plan.tsv", "\n".join(plan_lines) + "\n")
    _write(root / "run_index.tsv", "\n".join(index_lines) + "\n")

    scratch = slurm.get("scratch", {})
    node_local = scratch.get("mode") == "node_local"
    array_template = _select_array_template(config)
    generated = root / "generated_slurm"
    script_names: dict[str, str] = {}

    if scratch.get("mode") == "persistent_cache":
        paths = slurm.get("paths", {})
        environment = slurm.get("environment", {})
        setup = slurm.get("setup", {})
        cache_build_name = "cache_build_job.sh"
        cache_build_context = {
            "ACCOUNT": shell_quote(slurm.get("account", "")),
            "CACHE_ROOT": shell_quote(paths.get("cache_root", "cache/images")),
            "CONFIG_PATH": shell_quote(root.resolve() / "resolved_submission_config.yaml"),
            "CONDA_ENV": shell_quote(environment.get("conda_env", "fishspecies")),
            "CONDA_SH": shell_quote(environment.get("conda_sh", "")),
            "CPUS_PER_TASK": shell_quote(setup.get("cpus_per_task", 8)),
            "DATA_ROOT": shell_quote(paths.get("data_root", "data")),
            "MEMORY": shell_quote(setup.get("memory", "16G")),
            "METADATA_CSV": shell_quote(paths.get("metadata_csv", "metadata.csv")),
            "PROJECT_ROOT": shell_quote(paths.get("project_root", ".")),
            "TIME_LIMIT": shell_quote(setup.get("time_limit", "02:00:00")),
        }
        _write(
            generated / cache_build_name,
            render_template("cache_build_job.sh.tmpl", cache_build_context),
            0o755,
        )

    array_context = _array_context(
        plan,
        config,
        root.resolve(),
        node_local=node_local,
    )
    if node_local:
        monitoring = slurm.get("monitoring", {}) or {}
        array_context.update(
            {
                "MONITORING_ENABLED": (
                    "true" if bool(monitoring.get("enabled", False)) else "false"
                ),
                "MONITORING_INTERVAL": shell_quote(
                    monitoring.get("interval_seconds", 5)
                ),
                "SETUP_READY_MARKER": shell_quote(
                    f"{str(scratch.get('root')).rstrip('/')}/"
                    f"{scratch.get('ready_marker', 'READY')}"
                ),
            }
        )
    array_name = _array_output_name(array_template, config)
    _write(
        generated / array_name,
        render_template(array_template, array_context),
        0o755,
    )
    script_names["array"] = str(Path("generated_slurm") / array_name)

    if bool(slurm.get("setup", {}).get("enabled", False)):
        setup_context = {
            "DATA_ROOT": shell_quote(slurm.get("paths", {}).get("data_root", "data")),
            "DATA_COPY_COMMAND": shell_join(
                [
                    "rsync",
                    "-a",
                    "--include=*/",
                    *[
                        argument
                        for pattern in scratch.get("data_include", [])
                        for argument in (f"--include={pattern}",)
                    ],
                    "--exclude=*",
                    str(slurm.get("paths", {}).get("data_root", "data")).rstrip("/") + "/",
                    f"{str(scratch.get('root')).rstrip('/')}/data/",
                ]
            ),
            "PROJECT_ROOT": shell_quote(slurm.get("paths", {}).get("project_root", ".")),
            "READY_MARKER": shell_quote(scratch.get("ready_marker", "READY")),
            "SCRATCH_DATA": shell_quote(f"{str(scratch.get('root')).rstrip('/')}/data"),
            "SCRATCH_OUTPUTS": shell_quote(f"{str(scratch.get('root')).rstrip('/')}/outputs"),
            "SCRATCH_PROJECT": shell_quote(f"{str(scratch.get('root')).rstrip('/')}/project"),
            "SCRATCH_ROOT": shell_quote(scratch.get("root", "/tmp/fish_species")),
            "SUBMISSION_ID": shell_quote(scratch.get("submission_id", "")),
        }
        setup_name = "node_local_setup_job.sh"
        _write(
            generated / setup_name,
            render_template("node_local_setup_job.sh.tmpl", setup_context),
            0o755,
        )
        script_names["setup"] = str(Path("generated_slurm") / setup_name)

    if bool(slurm.get("cleanup", {}).get("enabled", False)):
        cleanup_name = "node_local_cleanup_job.sh"
        _write(
            generated / cleanup_name,
            render_template(
                "node_local_cleanup_job.sh.tmpl",
                {
                    "SCRATCH_ROOT": shell_quote(
                        scratch.get("root", "/tmp/fish_species")
                    ),
                    "SUBMISSION_ID": shell_quote(
                        scratch.get("submission_id", "")
                    ),
                },
            ),
            0o755,
        )
        script_names["cleanup"] = str(Path("generated_slurm") / cleanup_name)

    if bool(slurm.get("collection", {}).get("enabled", False)):
        collector_kind = _collector_kind(config)
        collector_command = shell_join(
            [
                "python",
                "-m",
                "fish_species.slurm",
                "collect",
                "--results-root",
                str(Path(plan.results_root).resolve()),
                "--kind",
                collector_kind,
            ]
        )
        collect_context = {
            "CONDA_ENV": shell_quote(slurm.get("environment", {}).get("conda_env", "fishspecies")),
            "CONDA_SH": shell_quote(slurm.get("environment", {}).get("conda_sh", "")),
            "PROJECT_ROOT": shell_quote(slurm.get("paths", {}).get("project_root", ".")),
            "RESULTS_ROOT": shell_quote(Path(plan.results_root).resolve()),
        }
        collect_name = "result_collector_job.sh"
        collector_template = "result_collector_job.sh.tmpl"
        collect_context["COLLECT_COMMAND"] = collector_command
        _write(
            generated / collect_name,
            render_template(collector_template, collect_context),
            0o755,
        )
        script_names["collect"] = str(Path("generated_slurm") / collect_name)

    jobs = _build_jobs(plan, config, script_names)
    _validate_job_graph(plan, jobs)
    log_root = root.resolve() / "slurm_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job["stdout"] = str(log_root / Path(job["stdout"]).name)
        job["stderr"] = str(log_root / Path(job["stderr"]).name)
    metadata = _execution_metadata(plan, config, jobs)
    submission_plan = {
        "schema_version": 1,
        "plan": plan.as_dict(),
        "metadata": metadata,
    }
    _write(root / "submission_plan.json", _json(submission_plan))
    manifest = {
        "schema_version": 1,
        "dry_run": True,
        "artifact_root": str(root.resolve()),
        "plan_sha256": plan.resolved_config_sha256,
        "array_size": plan.array_size,
        "metadata": metadata,
        "jobs": jobs,
    }
    _write(root / "submission_manifest.json", _json(manifest))
    _write(
        root / "dry_run.json",
        _json(
            {
                "submitted": False,
                "scheduler_calls": 0,
                "metadata": metadata,
            }
        ),
    )

    checksums = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_checksums.json":
            checksums[str(path.relative_to(root))] = {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
    _write(root / "artifact_checksums.json", _json(checksums))
    return manifest


def write_artifact_bundle(
    plan: SubmissionPlan,
    config: dict[str, Any],
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Atomically create one self-contained bundle without scheduler calls."""
    target = Path(artifact_dir).expanduser()
    if target.exists() or target.is_symlink():
        raise RenderError(f"Artifact directory already exists: {target}")
    target.parent.resolve().mkdir(parents=True, exist_ok=True)
    try:
        manifest = _render_bundle(plan, config, target)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return manifest


def verify_artifact_bundle(manifest_path: str | Path) -> dict[str, Any]:
    """Load a manifest only after verifying every recorded artifact hash."""
    path = Path(manifest_path).resolve()
    root = path.parent
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        checksums = json.loads(
            (root / "artifact_checksums.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RenderError(f"Invalid artifact bundle at {root}: {exc}") from exc
    for relative, expected in checksums.items():
        candidate = root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise RenderError(f"Artifact is missing or unsafe: {candidate}")
        if candidate.stat().st_size != expected["size"] or _sha256(candidate) != expected["sha256"]:
            raise RenderError(f"Artifact checksum mismatch: {candidate}")
    return manifest


__all__ = [
    "RenderError",
    "render_template",
    "shell_join",
    "shell_quote",
    "verify_artifact_bundle",
    "write_artifact_bundle",
]
