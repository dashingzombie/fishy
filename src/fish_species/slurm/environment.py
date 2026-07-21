"""Deterministic resolution of scheduler paths and historical environment aliases.

The canonical launcher is configuration-driven.  Historical shell variables are
therefore imported only when a caller explicitly opts in.  Resolution is pure:
it does not inspect the filesystem, create directories, or contact SLURM.
"""

from __future__ import annotations

import copy
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config.overrides import set_nested


class EnvironmentResolutionError(ValueError):
    """A declared environment value cannot be resolved safely."""


@dataclass(frozen=True)
class ResolutionContext:
    """Explicit process context used by the otherwise pure resolver."""

    cwd: Path
    environ: Mapping[str, str] = field(default_factory=dict)
    submission_stamp: str | None = None
    process_id: int | None = None


Parser = Callable[[str, str], Any]


@dataclass(frozen=True)
class LegacyEnvironmentBinding:
    """One allow-listed historical alias group and its canonical target."""

    names: tuple[str, ...]
    target: str
    parser: Parser


def _string(raw: str, name: str) -> str:
    if not raw:
        raise EnvironmentResolutionError(f"{name} must not be empty")
    return raw


def _optional_string(raw: str, _name: str) -> str:
    return raw


def _positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise EnvironmentResolutionError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise EnvironmentResolutionError(f"{name} must be a positive integer")
    return value


def _non_negative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise EnvironmentResolutionError(
            f"{name} must be a non-negative integer"
        ) from exc
    if value < 0:
        raise EnvironmentResolutionError(f"{name} must be a non-negative integer")
    return value


def _memory(raw: str, name: str) -> int | str:
    if raw.isdigit():
        return _positive_int(raw, name)
    if re.fullmatch(r"[1-9][0-9]*[KMGTkmgt]", raw):
        return raw
    raise EnvironmentResolutionError(
        f"{name} must be positive MiB or a K/M/G/T memory value"
    )


def _boolean(raw: str, name: str) -> bool:
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise EnvironmentResolutionError(f"{name} must be a boolean value")


def _copy_mode(raw: str, name: str) -> int | str:
    normalised = raw.strip().lower()
    if normalised == "auto":
        return "auto"
    if normalised in {"0", "1"}:
        return int(normalised)
    raise EnvironmentResolutionError(f"{name} must be auto, 0, or 1")


def _nodes(raw: str, name: str) -> list[str]:
    values = [item for item in re.split(r"[\s,]+", raw.strip()) if item]
    if not values:
        raise EnvironmentResolutionError(f"{name} must name at least one node")
    if len(values) != len(set(values)):
        raise EnvironmentResolutionError(f"{name} contains duplicate node names")
    return values


def _extra_args(raw: str, name: str) -> list[str]:
    try:
        values = shlex.split(raw)
    except ValueError as exc:
        raise EnvironmentResolutionError(f"{name} contains invalid shell quoting") from exc
    if any(not value.startswith("--") for value in values):
        raise EnvironmentResolutionError(
            f"{name} may contain only long-form sbatch arguments"
        )
    return values


# The order is stable and forms part of the snapshot-tested compatibility
# contract.  Aliases in one binding are equivalent; conflicting values fail
# rather than depending on arbitrary environment iteration order.
LEGACY_ENVIRONMENT_BINDINGS: tuple[LegacyEnvironmentBinding, ...] = (
    LegacyEnvironmentBinding(("SOURCE_ROOT", "PROJECT_SRC"), "slurm.paths.project_root", _string),
    LegacyEnvironmentBinding(("DATA_ROOT", "DATA_SRC"), "slurm.paths.data_root", _string),
    LegacyEnvironmentBinding(("METADATA_CSV",), "slurm.paths.metadata_csv", _string),
    LegacyEnvironmentBinding(("CACHE_DIR",), "slurm.paths.cache_root", _string),
    LegacyEnvironmentBinding(("RESULTS_ROOT",), "slurm.paths.results_root", _string),
    LegacyEnvironmentBinding(("SCRATCH_ROOT",), "slurm.scratch.root", _string),
    LegacyEnvironmentBinding(("CONDA_SH",), "slurm.environment.conda_sh", _string),
    LegacyEnvironmentBinding(("CONDA_ENV",), "slurm.environment.conda_env", _string),
    LegacyEnvironmentBinding(("GPU_ACCOUNT",), "slurm.account", _string),
    LegacyEnvironmentBinding(("GPU_PARTITION",), "slurm.partition", _string),
    LegacyEnvironmentBinding(("GPU_CPUS_PER_TASK",), "slurm.cpus_per_task", _positive_int),
    LegacyEnvironmentBinding(("GPU_MEM",), "slurm.memory", _memory),
    LegacyEnvironmentBinding(("GPU_TIME",), "slurm.time_limit", _string),
    LegacyEnvironmentBinding(("MAX_ACTIVE",), "slurm.array.max_active", _positive_int),
    LegacyEnvironmentBinding(("GPU_NODES",), "slurm.scratch.nodes", _nodes),
    LegacyEnvironmentBinding(
        ("COPY_CACHE_TO_TMP",), "slurm.scratch.copy_cache_to_tmp", _copy_mode
    ),
    LegacyEnvironmentBinding(
        ("TMP_RESERVE_GB",), "slurm.scratch.tmp_reserve_gb", _non_negative_int
    ),
    LegacyEnvironmentBinding(("SETUP_CPUS_PER_TASK",), "slurm.setup.cpus_per_task", _positive_int),
    LegacyEnvironmentBinding(("SETUP_MEM",), "slurm.setup.memory", _memory),
    LegacyEnvironmentBinding(("SETUP_TIME",), "slurm.setup.time_limit", _string),
    LegacyEnvironmentBinding(("CLEANUP_PARTITION",), "slurm.cleanup.partition", _string),
    LegacyEnvironmentBinding(
        ("CLEANUP_CPUS_PER_TASK",), "slurm.cleanup.cpus_per_task", _positive_int
    ),
    LegacyEnvironmentBinding(("CLEANUP_MEM",), "slurm.cleanup.memory", _memory),
    LegacyEnvironmentBinding(("CLEANUP_TIME",), "slurm.cleanup.time_limit", _string),
    LegacyEnvironmentBinding(("COLLECT_PARTITION",), "slurm.collection.partition", _string),
    LegacyEnvironmentBinding(
        ("COLLECT_CPUS_PER_TASK",), "slurm.collection.cpus_per_task", _positive_int
    ),
    LegacyEnvironmentBinding(("COLLECT_MEM",), "slurm.collection.memory", _memory),
    LegacyEnvironmentBinding(("COLLECT_TIME",), "slurm.collection.time_limit", _string),
    LegacyEnvironmentBinding(
        ("GPU_EXTRA_SBATCH_ARGS",),
        "slurm.submission.extra_sbatch_args",
        _extra_args,
    ),
    LegacyEnvironmentBinding(("WANDB_ENABLED",), "wandb.enabled", _boolean),
    LegacyEnvironmentBinding(("WANDB_PROJECT",), "wandb.project", _string),
    LegacyEnvironmentBinding(("WANDB_ENTITY",), "wandb.entity", _optional_string),
    LegacyEnvironmentBinding(("WANDB_MODE",), "wandb.mode", _string),
    LegacyEnvironmentBinding(("WANDB_RUN_GROUP",), "wandb.group", _optional_string),
)

_DEPRECATED_VARIABLES = (
    "BASE_CONFIG",
    "TRAIN_SCRIPT",
    "RUN_SPEC_GENERATOR",
    "RESULT_COLLECTOR",
    "COLLECT_EXTRA_SBATCH_ARGS",
    "CLEANUP_EXTRA_SBATCH_ARGS",
)

_PATH_TARGETS = (
    "slurm.scratch.root",
    "slurm.environment.conda_sh",
    "slurm.paths.project_root",
    "slurm.paths.data_root",
    "slurm.paths.metadata_csv",
    "slurm.paths.results_root",
    "slurm.paths.cache_root",
    "slurm.logging.directory",
)

_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def _get_nested(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _expand_path(raw: str, context: ResolutionContext, target: str) -> str:
    environment = context.environ

    def substitute(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("plain")
        if name not in environment:
            raise EnvironmentResolutionError(
                f"{target} references unavailable environment variable {name}"
            )
        return environment[name]

    value = _VARIABLE.sub(substitute, raw)
    if value == "~" or value.startswith("~/"):
        home = environment.get("HOME")
        if not home:
            raise EnvironmentResolutionError(
                f"{target} uses ~ but HOME is unavailable"
            )
        value = home + value[1:]
    return value


def _selected_environment(
    binding: LegacyEnvironmentBinding, environ: Mapping[str, str]
) -> tuple[str, str] | None:
    present = [(name, environ[name]) for name in binding.names if name in environ]
    if not present:
        return None
    distinct = {value for _, value in present}
    if len(distinct) > 1:
        names = ", ".join(name for name, _ in present)
        raise EnvironmentResolutionError(
            f"Conflicting legacy aliases for {binding.target}: {names}"
        )
    return present[0]


@dataclass(frozen=True)
class ResolvedEnvironment:
    """Resolved configuration plus deterministic compatibility diagnostics."""

    config: dict[str, Any]
    provenance: Mapping[str, str]
    imported_variables: tuple[str, ...]
    warnings: tuple[str, ...]

    def snapshot(self) -> dict[str, Any]:
        slurm = self.config.get("slurm", {}) or {}
        return {
            "cluster_profile": slurm.get("cluster_profile"),
            "paths": dict(slurm.get("paths", {}) or {}),
            "environment": dict(slurm.get("environment", {}) or {}),
            "resources": {
                key: slurm.get(key)
                for key in (
                    "account",
                    "partition",
                    "cpus_per_task",
                    "memory",
                    "time_limit",
                )
            },
            "array": dict(slurm.get("array", {}) or {}),
            "scratch": dict(slurm.get("scratch", {}) or {}),
            "setup": dict(slurm.get("setup", {}) or {}),
            "collection": dict(slurm.get("collection", {}) or {}),
            "cleanup": dict(slurm.get("cleanup", {}) or {}),
            "wandb": dict(self.config.get("wandb", {}) or {}),
            "imported_variables": list(self.imported_variables),
            "provenance": dict(sorted(self.provenance.items())),
            "warnings": list(self.warnings),
        }


def resolve_submission_environment(
    config: Mapping[str, Any],
    context: ResolutionContext,
    *,
    import_legacy: bool = False,
) -> ResolvedEnvironment:
    """Resolve declared paths and, when requested, historical env aliases."""
    resolved = copy.deepcopy(dict(config))
    provenance: dict[str, str] = {}
    imported: list[str] = []
    warnings: list[str] = []

    if import_legacy:
        # Genome historically treated PROJECT_ROOT as a symlinkable repository
        # entry, then derived separate source and data roots from its real path.
        project_entry = context.environ.get("PROJECT_ROOT")
        project_provenance = "legacy:PROJECT_ROOT"
        cluster_profile = _get_nested(resolved, "slurm.cluster_profile")
        if project_entry is None and cluster_profile == "genome":
            home = context.environ.get("HOME")
            if not home:
                raise EnvironmentResolutionError(
                    "Genome legacy resolution requires HOME or PROJECT_ROOT"
                )
            project_entry = str(Path(home) / "fish-species")
            project_provenance = "legacy-default:HOME"
        direct_source = any(
            name in context.environ for name in ("SOURCE_ROOT", "PROJECT_SRC")
        )
        direct_data = any(
            name in context.environ for name in ("DATA_ROOT", "DATA_SRC")
        )
        if project_entry is not None:
            expanded = _expand_path(project_entry, context, "PROJECT_ROOT")
            entry_path = Path(expanded)
            if not entry_path.is_absolute():
                entry_path = context.cwd / entry_path
            real_entry = entry_path.resolve(strict=False)
            if "PROJECT_ROOT" in context.environ:
                imported.append("PROJECT_ROOT")
            if not direct_source:
                set_nested(resolved, "slurm.paths.project_root", str(real_entry / "source"))
                provenance["slurm.paths.project_root"] = project_provenance
            if not direct_data:
                set_nested(resolved, "slurm.paths.data_root", str(real_entry / "data"))
                provenance["slurm.paths.data_root"] = project_provenance
                if "METADATA_CSV" not in context.environ:
                    set_nested(
                        resolved,
                        "slurm.paths.metadata_csv",
                        str(real_entry / "data" / "label_train.json"),
                    )
                    provenance["slurm.paths.metadata_csv"] = project_provenance
                if "CACHE_DIR" not in context.environ:
                    set_nested(
                        resolved,
                        "slurm.paths.cache_root",
                        str(real_entry / "data" / "image_cache"),
                    )
                    provenance["slurm.paths.cache_root"] = project_provenance

        for binding in LEGACY_ENVIRONMENT_BINDINGS:
            selected = _selected_environment(binding, context.environ)
            if selected is None:
                continue
            name, raw = selected
            set_nested(resolved, binding.target, binding.parser(raw, name))
            provenance[binding.target] = f"legacy:{name}"
            imported.append(name)

        data_name = next(
            (name for name in ("DATA_ROOT", "DATA_SRC") if name in context.environ),
            None,
        )
        if data_name is not None:
            data_root = Path(
                _expand_path(context.environ[data_name], context, data_name)
            )
            if "METADATA_CSV" not in context.environ:
                set_nested(
                    resolved,
                    "slurm.paths.metadata_csv",
                    str(data_root / "label_train.json"),
                )
                provenance["slurm.paths.metadata_csv"] = f"legacy:{data_name}"
            if "CACHE_DIR" not in context.environ:
                set_nested(
                    resolved,
                    "slurm.paths.cache_root",
                    str(data_root / "image_cache"),
                )
                provenance["slurm.paths.cache_root"] = f"legacy:{data_name}"

        if (
            cluster_profile == "genome"
            and "RESULTS_ROOT" not in context.environ
            and context.submission_stamp
        ):
            source_root = _get_nested(resolved, "slurm.paths.project_root")
            if not isinstance(source_root, str) or not source_root:
                raise EnvironmentResolutionError(
                    "Genome legacy result resolution requires a source root"
                )
            source_root = _expand_path(
                source_root, context, "slurm.paths.project_root"
            )
            set_nested(
                resolved,
                "slurm.paths.results_root",
                str(
                    Path(source_root)
                    / "outputs_slurm"
                    / f"persistent_cache_sweep_{context.submission_stamp}"
                ),
            )
            provenance["slurm.paths.results_root"] = provenance.get(
                "slurm.paths.project_root", project_provenance
            )

        # GHPC launchers used the current project directory as their entry and
        # generated unique result/scratch roots from one timestamp and PID.
        if "PROJECT_SRC" in context.environ and context.submission_stamp:
            project_src = Path(
                _expand_path(context.environ["PROJECT_SRC"], context, "PROJECT_SRC")
            )
            if not project_src.is_absolute():
                project_src = context.cwd / project_src
            if "RESULTS_ROOT" not in context.environ:
                set_nested(
                    resolved,
                    "slurm.paths.results_root",
                    str(
                        project_src
                        / "outputs_slurm"
                        / f"node_local_sweep_{context.submission_stamp}"
                    ),
                )
                provenance["slurm.paths.results_root"] = "legacy:PROJECT_SRC"
            if "SCRATCH_ROOT" not in context.environ:
                scratch_id = context.environ.get("SCRATCH_ID")
                if not scratch_id:
                    suffix = (
                        f"_{context.process_id}"
                        if context.process_id is not None
                        else ""
                    )
                    scratch_id = (
                        f"fish_node_local_sweep_{context.submission_stamp}{suffix}"
                    )
                user = context.environ.get("USER")
                if not user:
                    raise EnvironmentResolutionError(
                        "GHPC legacy scratch resolution requires USER"
                    )
                set_nested(
                    resolved,
                    "slurm.scratch.root",
                    f"/scratch/{user}/{scratch_id}",
                )
                provenance["slurm.scratch.root"] = "legacy:PROJECT_SRC"

        for name in _DEPRECATED_VARIABLES:
            if name in context.environ:
                imported.append(name)
                warnings.append(
                    f"{name} is ignored by the canonical launcher; use its CLI/config equivalent"
                )

    scratch = _get_nested(resolved, "slurm.scratch")
    if (
        isinstance(scratch, dict)
        and scratch.get("mode") == "node_local"
        and bool(scratch.get("unique_per_submission", False))
    ):
        if not context.submission_stamp or context.process_id is None:
            raise EnvironmentResolutionError(
                "node-local scratch resolution requires a submission timestamp and PID"
            )
        submission_id = f"{context.submission_stamp}_{context.process_id}"
        raw_root = scratch.get("root")
        if not isinstance(raw_root, str) or not raw_root:
            raise EnvironmentResolutionError(
                "node-local scratch requires a non-empty root prefix"
            )
        if "{submission_id}" in raw_root:
            resolved_root = raw_root.replace("{submission_id}", submission_id)
        elif submission_id in raw_root:
            resolved_root = raw_root
        else:
            resolved_root = f"{raw_root.rstrip('_-')}_{submission_id}"
        scratch["root"] = resolved_root
        scratch["submission_id"] = submission_id
        provenance["slurm.scratch.root"] = "generated:submission_id"
        provenance["slurm.scratch.submission_id"] = "generated:timestamp+pid"

    for target in _PATH_TARGETS:
        value = _get_nested(resolved, target)
        if isinstance(value, str):
            set_nested(resolved, target, _expand_path(value, context, target))
            provenance.setdefault(target, "config")

    return ResolvedEnvironment(
        config=resolved,
        provenance=provenance,
        imported_variables=tuple(dict.fromkeys(imported)),
        warnings=tuple(warnings),
    )


__all__ = [
    "EnvironmentResolutionError",
    "LEGACY_ENVIRONMENT_BINDINGS",
    "LegacyEnvironmentBinding",
    "ResolutionContext",
    "ResolvedEnvironment",
    "resolve_submission_environment",
]
