"""Pure planning, persistence, collection, and ranking for phased sweeps."""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
import re
from contextlib import contextmanager
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from ..config import load_config, set_nested, validate_config
from .schema import PipelineConfigError


class PipelineError(RuntimeError):
    """A pipeline operation could not be completed safely."""


_EPHEMERAL_KEYS = {
    "artifact_root", "created_at", "generated_at", "hostname", "host",
    "job_id", "job_ids", "log_dir", "logs", "pid", "process_id",
    "slurm_job_id", "slurm_job_ids", "submission_stamp", "submitted_at",
    "timestamp", "wandb_run_id",
}
_EPHEMERAL_TOP_LEVEL = {"output", "pipeline_run", "slurm"}
_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _scientific_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for child_key in sorted(value):
            if child_key in _EPHEMERAL_KEYS:
                continue
            if key is None and child_key in _EPHEMERAL_TOP_LEVEL:
                continue
            if key == "wandb" and child_key in {"id", "name", "group"}:
                continue
            if key == "fine_tuning" and child_key == "checkpoint_path":
                continue
            cleaned[str(child_key)] = _scientific_value(
                value[child_key], key=str(child_key)
            )
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_scientific_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def scientific_config_hash(config: Mapping[str, Any]) -> str:
    """Hash resolved scientific settings while excluding execution metadata."""
    payload = json.dumps(
        _scientific_value(config), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pipeline_hash(document: Mapping[str, Any]) -> str:
    """Return the immutable definition hash for a loaded pipeline document."""
    value = copy.deepcopy(dict(document))
    value.pop("_pipeline_file", None)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resolve_pipeline_path(document: Mapping[str, Any], value: str) -> Path:
    """Resolve a repository-relative or pipeline-file-relative path."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists() or value.startswith("outputs/") or value.startswith("logs/"):
        return cwd_candidate
    source = Path(str(document.get("_pipeline_file", Path.cwd()))).resolve()
    return (source.parent / candidate).resolve()


def pipeline_paths(document: Mapping[str, Any]) -> tuple[Path, Path]:
    """Return output-root and state paths."""
    pipeline = document["pipeline"]
    root = resolve_pipeline_path(document, str(pipeline["output_root"]))
    configured = pipeline.get("state_file")
    state = resolve_pipeline_path(document, str(configured)) if configured else root / "state.json"
    return root, state


def phase_by_name(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Look up one configured phase by its unique name."""
    for phase in document["pipeline"]["phases"]:
        if phase["name"] == name:
            return phase
    raise PipelineError(f"unknown pipeline phase: {name}")


def effective_selection(document: Mapping[str, Any], phase: Mapping[str, Any]) -> dict[str, Any]:
    """Merge global and phase-specific ranking settings."""
    result = copy.deepcopy(document["pipeline"].get("selection", {}))
    result.update(copy.deepcopy(phase.get("selection", {}) or {}))
    result.setdefault("direction", "max")
    result.setdefault("top_k", 1)
    result.setdefault("tie_tolerance", 1e-12)
    result.setdefault("tie_breakers", [])
    result.setdefault("constraints", [])
    return result


def _phase_variants(phase: Mapping[str, Any]) -> list[tuple[str, dict[str, Any], bool]]:
    phase_type = str(phase["type"])
    if phase_type == "cartesian":
        parameters = phase["parameters"]
        keys = list(parameters)
        variants = []
        for index, values in enumerate(itertools.product(*(parameters[key] for key in keys))):
            variants.append((f"combination_{index:03d}", dict(zip(keys, values)), False))
        return variants
    conditions = phase.get("conditions")
    if isinstance(conditions, list) and conditions:
        return [
            (str(item["name"]), copy.deepcopy(item.get("overrides", {})), bool(item.get("baseline", False)))
            for item in conditions
        ]
    return [(phase_type, {}, False)]


def estimated_phase_counts(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Estimate run counts assuming every selection produces its configured top-k."""
    selected: dict[str, int] = {"base": 1}
    result = []
    for phase in document["pipeline"]["phases"]:
        parents = 1 if phase.get("parent", "base") == "base" else int(
            phase.get("inherit_top_k", selected[phase["parent"]])
        )
        variants = len(_phase_variants(phase))
        count = parents * variants
        selected[phase["name"]] = min(count, int(effective_selection(document, phase)["top_k"]))
        result.append({
            "phase": phase["name"], "parents": parents,
            "variants": variants, "runs": count, "type": phase["type"],
        })
    return result


def _apply_dotted(config: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:
            raise PipelineError("phase override keys must be non-empty strings")
        set_nested(config, key, copy.deepcopy(value))


def _slug(value: str) -> str:
    return _SLUG.sub("-", value).strip("-._") or "run"


def _normalise_parent(parent: Mapping[str, Any] | None, base: dict[str, Any]) -> dict[str, Any]:
    if parent is None:
        return {
            "run_id": None, "configuration_hash": scientific_config_hash(base),
            "checkpoint": None, "metrics": {}, "resolved_config": base,
        }
    required = {"run_id", "configuration_hash", "resolved_config"}
    missing = required - set(parent)
    if missing:
        raise PipelineError(f"parent record missing: {', '.join(sorted(missing))}")
    return dict(parent)


def expand_phase(
    document: Mapping[str, Any],
    phase: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand one phase, inheriting each complete selected parent configuration."""
    base_path = resolve_pipeline_path(document, str(document["pipeline"]["base_config"]))
    base = load_config(base_path)
    parent_items: Sequence[Mapping[str, Any] | None] = list(parents or [None])
    variants = _phase_variants(phase)
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for parent_raw in parent_items:
        parent = _normalise_parent(parent_raw, base)
        for variant_name, variant_overrides, baseline in variants:
            config = copy.deepcopy(parent["resolved_config"])
            phase_overrides: dict[str, Any] = {}
            for source in (
                phase.get("overrides", {}) or {},
                phase.get("common_overrides", {}) or {},
                variant_overrides,
            ):
                phase_overrides.update(copy.deepcopy(source))
                _apply_dotted(config, source)
            set_nested(config, "sweep.enabled", False)
            set_nested(config, "matched_condition_training.enabled", False)
            phase_type = str(phase["type"])
            parent_checkpoint = parent.get("checkpoint")
            if phase_type in {"evaluation_only", "fine_tune"}:
                if not parent_checkpoint:
                    raise PipelineError(
                        f"phase {phase['name']!r} requires a selected parent checkpoint"
                    )
                set_nested(config, "fine_tuning.enabled", True)
                set_nested(config, "fine_tuning.checkpoint_path", str(parent_checkpoint))
                set_nested(config, "fine_tuning.reset_optimizer", True)
            if phase_type == "evaluation_only":
                mode = "evaluation_only"
            else:
                mode = "train"
            digest = scientific_config_hash(config)
            if digest in seen:
                raise PipelineError(
                    f"phase {phase['name']!r} creates duplicate scientific configurations "
                    f"for {seen[digest]!r} and {variant_name!r}"
                )
            seen[digest] = variant_name
            run_id = f"{_slug(str(phase['name']))}-{_slug(variant_name)}-{digest[:12]}"
            output_root, _ = pipeline_paths(document)
            phase_root = output_root / "phases" / str(phase["name"])
            output_dir = phase_root / "runs" / run_id
            expected_checkpoint = (
                str(parent_checkpoint) if mode == "evaluation_only"
                else str(output_dir / "best_model.pt")
            )
            set_nested(config, "output.out_dir", str(phase_root / "runs"))
            config["pipeline_run"] = {
                "run_id": run_id,
                "configuration_hash": digest,
                "phase": phase["name"],
                "parent_phase": phase.get("parent", "base"),
                "parent_run_id": parent.get("run_id"),
                "parent_configuration_hash": parent.get("configuration_hash"),
                "parent_checkpoint": parent_checkpoint,
                "inherited_metrics": copy.deepcopy(parent.get("metrics", {})),
                "phase_overrides": copy.deepcopy(phase_overrides),
                "execution_mode": mode,
            }
            records.append({
                "run_index": len(records), "slurm_array_index": len(records),
                "run_id": run_id, "configuration_hash": digest,
                "resolved_config": config, "resolved_config_path": None,
                "parent_phase": phase.get("parent", "base"),
                "parent_run_id": parent.get("run_id"),
                "parent_configuration_hash": parent.get("configuration_hash"),
                "parent_checkpoint": parent_checkpoint,
                "inherited_metrics": copy.deepcopy(parent.get("metrics", {})),
                "phase_overrides": phase_overrides, "condition": variant_name,
                "baseline": baseline, "expected_output_directory": str(output_dir),
                "expected_checkpoint_path": expected_checkpoint,
                "submission_status": "planned", "completion_status": "pending",
                "retry_count": 0, "failure_category": None,
            })
    return records


def validate_generated_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Run canonical training validation over all generated configurations."""
    for record in records:
        try:
            validate_config(
                dict(record["resolved_config"]), workflow="training",
                check_paths=False, check_model_registry=False,
            )
        except Exception as exc:
            raise PipelineConfigError(
                f"generated run {record['run_id']} is invalid: {exc}"
            ) from exc
        image_size = int(record["resolved_config"].get("preprocessing", {}).get("image_size", 224))
        model = str(record["resolved_config"].get("model", {}).get("name", ""))
        if "dinov3" in model and "vit" in model and image_size % 16:
            raise PipelineConfigError(
                f"DINOv3 ViT image size must be divisible by 16, got {image_size}"
            )


def validate_pipeline_semantics(document: Mapping[str, Any]) -> dict[str, int]:
    """Validate every phase along a deterministic top-k-sized parent frontier."""
    frontiers: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for phase in document["pipeline"]["phases"]:
        parent_name = str(phase.get("parent", "base"))
        parents = None if parent_name == "base" else frontiers[parent_name][
            : int(phase.get("inherit_top_k", len(frontiers[parent_name])))
        ]
        records = expand_phase(document, phase, parents)
        validate_generated_records(records)
        counts[str(phase["name"])] = len(records)
        top_k = int(effective_selection(document, phase)["top_k"])
        frontiers[str(phase["name"])] = [
            {
                "run_id": record["run_id"],
                "configuration_hash": record["configuration_hash"],
                "checkpoint": record["expected_checkpoint_path"],
                "metrics": {},
                "resolved_config": record["resolved_config"],
            }
            for record in records[:top_k]
        ]
    return counts


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    """Persist JSON with fsync and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def state_lock(state_path: Path) -> Iterator[None]:
    """Serialize state-changing CLI operations on shared filesystems."""
    lock = state_path.with_suffix(state_path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initial_state(document: Mapping[str, Any]) -> dict[str, Any]:
    """Build the persistent state skeleton."""
    phases = {
        phase["name"]: {
            "status": "not_planned", "phase_type": phase["type"],
            "slurm_job_ids": [], "collector_job_ids": [], "planned_runs": 0,
            "running_runs": 0, "successful_runs": 0, "failed_runs": 0,
            "missing_runs": 0, "selected_run_ids": [], "best_metric": None,
            "attempts": 0, "runs": {},
        }
        for phase in document["pipeline"]["phases"]
    }
    return {
        "schema_version": 1, "pipeline_name": document["pipeline"]["name"],
        "pipeline_hash": pipeline_hash(document), "status": "planned",
        "current_phase": document["pipeline"]["phases"][0]["name"],
        "phases": phases,
    }


def load_state(document: Mapping[str, Any], *, required: bool = True) -> dict[str, Any] | None:
    """Load state and reject accidental use with a changed pipeline definition."""
    _, path = pipeline_paths(document)
    if not path.exists():
        if required:
            raise PipelineError(f"pipeline state does not exist: {path}")
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid pipeline state: {path}: {exc}") from exc
    if state.get("pipeline_hash") != pipeline_hash(document):
        raise PipelineError("pipeline definition changed since state was created")
    return state


def write_state(document: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Atomically save state at its configured path."""
    _, path = pipeline_paths(document)
    atomic_write_json(path, state)


def materialize_phase(
    document: Mapping[str, Any], phase: Mapping[str, Any], records: list[dict[str, Any]],
) -> Path:
    """Write immutable resolved configs and manifest JSON/JSONL for one phase."""
    root, _ = pipeline_paths(document)
    phase_root = root / "phases" / str(phase["name"])
    configs = phase_root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    public_records: list[dict[str, Any]] = []
    for record in records:
        path = configs / f"{record['run_id']}.yaml"
        payload = yaml.safe_dump(record["resolved_config"], sort_keys=False)
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise PipelineError(f"immutable resolved config changed: {path}")
        if not path.exists():
            _atomic_text(path, payload)
        record["resolved_config_path"] = str(path)
        public_records.append({key: copy.deepcopy(value) for key, value in record.items() if key != "resolved_config"})
    manifest = {
        "schema_version": 1, "pipeline_name": document["pipeline"]["name"],
        "pipeline_hash": pipeline_hash(document), "phase": phase["name"],
        "phase_type": phase["type"], "runs": public_records,
    }
    manifest_path = phase_root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = lambda value: [
            {k: v for k, v in row.items() if k not in {"submission_status", "completion_status", "retry_count", "failure_category"}}
            for row in value.get("runs", [])
        ]
        if old.get("pipeline_hash") != manifest["pipeline_hash"] or immutable(old) != immutable(manifest):
            raise PipelineError(f"immutable manifest changed: {manifest_path}")
    else:
        atomic_write_json(manifest_path, manifest)
        _atomic_text(
            phase_root / "manifest.jsonl",
            "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in public_records),
        )
    return manifest_path


def load_manifest(document: Mapping[str, Any], phase_name: str) -> dict[str, Any]:
    """Load one immutable phase manifest."""
    root, _ = pipeline_paths(document)
    path = root / "phases" / phase_name / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise PipelineError(f"phase manifest unavailable: {path}") from exc
    if manifest.get("pipeline_hash") != pipeline_hash(document):
        raise PipelineError(f"phase manifest hash mismatch: {phase_name}")
    return manifest


def parent_records(document: Mapping[str, Any], state: Mapping[str, Any], phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Load complete resolved configs and results for selected parent runs."""
    parent_name = str(phase.get("parent", "base"))
    if parent_name == "base":
        return []
    selected = list(state["phases"][parent_name].get("selected_run_ids", []))
    limit = int(phase.get("inherit_top_k", len(selected)))
    selected = selected[:limit]
    if not selected:
        raise PipelineError(f"phase {phase['name']!r} has no selected parents")
    manifest = load_manifest(document, parent_name)
    by_id = {item["run_id"]: item for item in manifest["runs"]}
    result = []
    for run_id in selected:
        item = by_id.get(run_id)
        if item is None:
            raise PipelineError(f"selected parent {run_id!r} is absent from manifest")
        config = load_config(item["resolved_config_path"])
        dynamic = state["phases"][parent_name]["runs"].get(run_id, {})
        result.append({
            "run_id": run_id, "configuration_hash": item["configuration_hash"],
            "checkpoint": dynamic.get("checkpoint", item["expected_checkpoint_path"]),
            "metrics": dynamic.get("metrics", {}), "resolved_config": config,
        })
    return result


def plan_phase(document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand, validate, materialize, and register one phase idempotently."""
    existing = state["phases"][phase["name"]]
    if existing["status"] != "not_planned":
        manifest = load_manifest(document, phase["name"])
        records = []
        for public in manifest["runs"]:
            item = copy.deepcopy(public)
            item["resolved_config"] = load_config(item["resolved_config_path"])
            records.append(item)
        return records
    parents = parent_records(document, state, phase) if phase.get("parent", "base") != "base" else None
    records = expand_phase(document, phase, parents)
    validate_generated_records(records)
    materialize_phase(document, phase, records)
    existing.update({
        "status": "planned", "planned_runs": len(records),
        "runs": {
            item["run_id"]: {
                "submission_status": "planned", "completion_status": "pending",
                "retry_count": 0, "failure_category": None, "metrics": {},
                "checkpoint": None,
            }
            for item in records
        },
    })
    state["current_phase"] = phase["name"]
    return records


def _finite_metrics(value: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        numeric = float(raw)
        if math.isfinite(numeric):
            result[str(key)] = numeric
    return result


def _failure_category(status: Mapping[str, Any] | None, missing: bool) -> tuple[str, bool]:
    if missing:
        return "missing_output", True
    text = " ".join(str(status.get(key, "")) for key in ("failure_category", "error", "reason", "slurm_state")).lower()
    if "preempt" in text:
        return "preempted", True
    if any(word in text for word in ("node_fail", "infrastructure", "timeout", "filesystem", "network")):
        return "infrastructure", True
    if "out of memory" in text or "cuda oom" in text:
        return "cuda_oom", False
    if "dataset" in text or "no such file" in text:
        return "missing_dataset", False
    if "config" in text or "incompatible" in text:
        return "configuration", False
    return "training_failure", False


def retry_decision(
    run: Mapping[str, Any], execution: Mapping[str, Any]
) -> tuple[bool, str]:
    """Return whether one failed run is eligible for automatic retry."""
    if not bool(execution.get("retry_failed", False)):
        return False, "automatic retries disabled"
    if int(run.get("retry_count", 0)) >= int(execution.get("maximum_retries", 0)):
        return False, "retry limit reached"
    category = str(run.get("failure_category") or "")
    if category not in {"missing_output", "infrastructure", "preempted"}:
        return False, f"failure category {category or 'unknown'} is not retryable"
    return True, "eligible"


def collect_local_results(
    document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect strict local result/status files and update dynamic run state."""
    manifest = load_manifest(document, str(phase["name"]))
    results_cfg = document["pipeline"].get("results", {}) or {}
    metric_file = str(results_cfg.get("metric_file", "metrics/validation_summary.json"))
    required_status = str(results_cfg.get("require_status", "completed"))
    collected: list[dict[str, Any]] = []
    counts = {"successful": 0, "failed": 0, "missing": 0, "running": 0}
    for item in manifest["runs"]:
        output = Path(item["expected_output_directory"])
        status_path = output / "run_status.json"
        dynamic = state["phases"][phase["name"]]["runs"][item["run_id"]]
        if not status_path.exists():
            category, _ = _failure_category(None, True)
            dynamic.update({"completion_status": "missing", "failure_category": category})
            counts["missing"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "run status is missing"})
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            dynamic.update({"completion_status": "failed", "failure_category": "invalid_status"})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "run status is invalid JSON"})
            continue
        if status.get("status") in {"running", "started", "submitted"}:
            dynamic["completion_status"] = "running"
            counts["running"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "run is incomplete"})
            continue
        if status.get("status") != required_status or int(status.get("exit_code", 1)) != 0:
            category, _ = _failure_category(status, False)
            dynamic.update({"completion_status": "failed", "failure_category": category})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": f"run status is {status.get('status')!r}"})
            continue
        if status.get("configuration_hash") != item["configuration_hash"]:
            dynamic.update({"completion_status": "failed", "failure_category": "configuration_hash_mismatch"})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "configuration hash does not match manifest"})
            continue
        metrics_path = output / metric_file
        try:
            raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            dynamic.update({"completion_status": "failed", "failure_category": "invalid_metrics"})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "metrics are missing or invalid"})
            continue
        metrics = _finite_metrics(raw_metrics)
        selection = effective_selection(document, phase)
        primary = str(selection.get("primary_metric", ""))
        if primary not in metrics or not math.isfinite(float(raw_metrics.get(primary, math.nan))):
            dynamic.update({"completion_status": "failed", "failure_category": "invalid_metrics"})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": f"metric {primary!r} is missing or non-finite"})
            continue
        checkpoint = Path(str(status.get("best_checkpoint", item["expected_checkpoint_path"])))
        if not checkpoint.is_absolute():
            checkpoint = output / checkpoint
        if not checkpoint.is_file():
            dynamic.update({"completion_status": "failed", "failure_category": "missing_checkpoint"})
            counts["failed"] += 1
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "checkpoint is missing"})
            continue
        dynamic.update({
            "completion_status": "successful", "failure_category": None,
            "metrics": metrics, "checkpoint": str(checkpoint),
            "best_epoch": status.get("best_epoch"),
        })
        counts["successful"] += 1
        collected.append({**item, **dynamic, "valid_result": True, "rejection_reason": None})
    phase_state = state["phases"][phase["name"]]
    for key in counts:
        phase_state[f"{key}_runs"] = counts[key]
    return collected


def collect_wandb_results(
    document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect W&B summaries matched by immutable configuration hash."""
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise PipelineError("W&B result collection requires the wandb package") from exc
    cfg = document["pipeline"]["results"]
    runs = wandb.Api().runs(f"{cfg['entity']}/{cfg['project']}")
    by_hash = {}
    for run in runs:
        run_cfg = dict(run.config or {})
        marker = run_cfg.get("pipeline_run", {}) or {}
        digest = marker.get("configuration_hash") or run_cfg.get("configuration_hash")
        if digest:
            by_hash[str(digest)] = run
    manifest = load_manifest(document, str(phase["name"]))
    collected = []
    phase_state = state["phases"][phase["name"]]
    for item in manifest["runs"]:
        dynamic = phase_state["runs"][item["run_id"]]
        run = by_hash.get(item["configuration_hash"])
        if run is None:
            dynamic.update({"completion_status": "missing", "failure_category": "missing_output"})
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "W&B run is missing"})
            continue
        metrics = _finite_metrics(dict(run.summary or {}))
        primary = effective_selection(document, phase).get("primary_metric")
        if run.state != "finished" or primary not in metrics:
            dynamic.update({"completion_status": "failed", "failure_category": "invalid_metrics"})
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "W&B run is incomplete or metric is absent"})
            continue
        checkpoint = str(dict(run.summary or {}).get("best_checkpoint", item["expected_checkpoint_path"]))
        if not Path(checkpoint).is_file():
            dynamic.update({"completion_status": "failed", "failure_category": "missing_checkpoint"})
            collected.append({**item, **dynamic, "valid_result": False, "rejection_reason": "checkpoint is missing"})
            continue
        dynamic.update({"completion_status": "successful", "failure_category": None, "metrics": metrics, "checkpoint": checkpoint})
        collected.append({**item, **dynamic, "valid_result": True, "rejection_reason": None})
    for status in ("successful", "failed", "missing", "running"):
        phase_state[f"{status}_runs"] = sum(item.get("completion_status") == status for item in collected)
    return collected


def collect_results(document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect results using the configured source."""
    if document["pipeline"].get("results", {}).get("source", "local") == "wandb":
        return collect_wandb_results(document, state, phase)
    return collect_local_results(document, state, phase)


def _constraint_threshold(constraint: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> float:
    if constraint.get("value_from") == "phase_max":
        metric = str(constraint["metric"])
        values = [float(row["metrics"][metric]) for row in rows if metric in row.get("metrics", {})]
        if not values:
            raise PipelineError(f"constraint metric {metric!r} is unavailable")
        return max(values) * float(constraint.get("fraction", 1.0))
    return float(constraint["value"])


def _passes(value: float, operator: str, threshold: float) -> bool:
    return {
        "greater_equal": value >= threshold, "less_equal": value <= threshold,
        "greater": value > threshold, "less": value < threshold,
        "equal": value == threshold,
    }[operator]


def _compute_estimate(record: Mapping[str, Any]) -> float:
    config = load_config(record["resolved_config_path"])
    training = config.get("training", {}) or {}
    size = float((config.get("preprocessing", {}) or {}).get("image_size", 224))
    epochs = float(training.get("epochs", 1))
    stage2 = float((((config.get("long_tail", {}) or {}).get("staged_training", {}) or {}).get("stage2_epochs", 0)) or 0)
    return (epochs + stage2) * size * size


def rank_results(
    document: Mapping[str, Any], phase: Mapping[str, Any], rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply constraints and stable deterministic multi-metric ranking."""
    selection = effective_selection(document, phase)
    valid_results = [row for row in rows if row.get("valid_result")]
    baseline_constraints = [
        item for item in selection.get("constraints", [])
        if item.get("operator") == "drop_from_phase_baseline_less_equal"
    ]
    baseline_name = selection.get("baseline_condition")
    baselines_by_parent: dict[str, list[dict[str, Any]]] = {}
    for candidate in valid_results:
        is_baseline = (
            candidate.get("condition") == baseline_name
            if baseline_name else bool(candidate.get("baseline"))
        )
        if is_baseline:
            group = str(candidate.get("parent_run_id") or "base")
            baselines_by_parent.setdefault(group, []).append(candidate)
    if baseline_constraints:
        groups = {str(row.get("parent_run_id") or "base") for row in valid_results}
        invalid_groups = [group for group in groups if len(baselines_by_parent.get(group, [])) != 1]
        if invalid_groups:
            raise PipelineError(
                "baseline-relative constraints require exactly one baseline result "
                "per parent; invalid groups: " + ", ".join(sorted(invalid_groups))
            )
    for row in rows:
        row["constraint_status"] = "not_evaluated"
        row["raw_rank"] = None
        row["selected_rank"] = None
        if not row.get("valid_result"):
            continue
        reasons = []
        for constraint in selection.get("constraints", []):
            metric = str(constraint["metric"])
            if metric not in row.get("metrics", {}):
                reasons.append(f"constraint metric {metric} is missing")
                continue
            value = float(row["metrics"][metric])
            operator = str(constraint["operator"])
            if operator == "drop_from_phase_baseline_less_equal":
                group = str(row.get("parent_run_id") or "base")
                baseline_value = float(
                    baselines_by_parent[group][0]["metrics"].get(metric, math.nan)
                )
                passed = math.isfinite(baseline_value) and baseline_value - value <= float(constraint["value"])
            else:
                threshold = _constraint_threshold(constraint, valid_results)
                passed = _passes(value, operator, threshold)
            if not passed:
                reasons.append(f"constraint failed: {metric} {operator}")
        if reasons:
            row["valid_result"] = False
            row["constraint_status"] = "failed"
            row["rejection_reason"] = "; ".join(reasons)
        else:
            row["constraint_status"] = "passed"
        row["compute_estimate"] = _compute_estimate(row)

    metrics = [(str(selection["primary_metric"]), str(selection.get("direction", "max")))]
    metrics.extend((str(item["metric"]), str(item.get("direction", "max"))) for item in selection.get("tie_breakers", []))
    tolerance = float(selection.get("tie_tolerance", 1e-12))

    def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
        if bool(left.get("valid_result")) != bool(right.get("valid_result")):
            return -1 if left.get("valid_result") else 1
        for metric, direction in metrics:
            lv = float(left.get("metrics", {}).get(metric, -math.inf if direction == "max" else math.inf))
            rv = float(right.get("metrics", {}).get(metric, -math.inf if direction == "max" else math.inf))
            if abs(lv - rv) > tolerance:
                if direction == "max":
                    return -1 if lv > rv else 1
                return -1 if lv < rv else 1
        lc = float(left.get("compute_estimate", math.inf))
        rc = float(right.get("compute_estimate", math.inf))
        if lc != rc:
            return -1 if lc < rc else 1
        lh, rh = str(left["configuration_hash"]), str(right["configuration_hash"])
        return -1 if lh < rh else (1 if lh > rh else 0)

    ranked = sorted(rows, key=cmp_to_key(compare))
    raw = 0
    selected = 0
    top_k = int(selection["top_k"])
    for row in ranked:
        if row.get("completion_status") == "successful":
            raw += 1
            row["raw_rank"] = raw
        if row.get("valid_result") and selected < top_k:
            selected += 1
            row["selected_rank"] = selected
    return ranked


def write_leaderboard(document: Mapping[str, Any], phase: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    """Write machine-readable and inspectable phase leaderboards."""
    root, _ = pipeline_paths(document)
    phase_root = root / "phases" / str(phase["name"])
    json_path = phase_root / "leaderboard.json"
    csv_path = phase_root / "leaderboard.csv"
    atomic_write_json(json_path, list(rows))
    fieldnames = [
        "raw_rank", "selected_rank", "run_id", "configuration_hash", "condition",
        "completion_status", "constraint_status", "rejection_reason", "compute_estimate", "metrics",
    ]
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if key == "metrics" else row.get(key) for key in fieldnames})
    os.replace(temporary, csv_path)
    return csv_path, json_path


def update_phase_ranking(document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank a collected phase and persist selected lineage in state."""
    ranked = rank_results(document, phase, rows)
    selected = [row for row in ranked if row.get("selected_rank") is not None]
    phase_state = state["phases"][phase["name"]]
    phase_state["selected_run_ids"] = [row["run_id"] for row in selected]
    primary = effective_selection(document, phase).get("primary_metric")
    phase_state["best_metric"] = selected[0]["metrics"].get(primary) if selected else None
    incomplete = any(row.get("completion_status") in {"pending", "running", "missing"} for row in ranked)
    phase_state["status"] = "partial" if incomplete else ("completed" if selected else "failed")
    write_leaderboard(document, phase, ranked)
    return ranked


def next_phase(document: Mapping[str, Any], current: str) -> dict[str, Any] | None:
    """Return the configured successor phase."""
    phases = document["pipeline"]["phases"]
    for index, phase in enumerate(phases):
        if phase["name"] == current:
            return phases[index + 1] if index + 1 < len(phases) else None
    raise PipelineError(f"unknown current phase: {current}")


def status_summary(document: Mapping[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build a concise status response for terminal or JSON output."""
    if state is None:
        return {"pipeline": document["pipeline"]["name"], "status": "not_started", "next_action": "submit or plan"}
    current = str(state["current_phase"])
    phases = []
    for phase in document["pipeline"]["phases"]:
        value = state["phases"][phase["name"]]
        phases.append({
            "name": phase["name"], "status": value["status"],
            "job_ids": value.get("slurm_job_ids", []),
            "planned": value.get("planned_runs", 0), "running": value.get("running_runs", 0),
            "successful": value.get("successful_runs", 0), "failed": value.get("failed_runs", 0),
            "missing": value.get("missing_runs", 0), "best_metric": value.get("best_metric"),
            "selected": value.get("selected_run_ids", []),
        })
    current_state = state["phases"][current]
    if current_state["status"] == "not_planned":
        action = f"submit phase {current}"
    elif current_state["status"] in {"submitted", "running"}:
        action = f"wait for or advance phase {current}"
    elif current_state["status"] in {"partial", "failed"}:
        action = f"resume phase {current}"
    else:
        successor = next_phase(document, current)
        action = "pipeline complete" if successor is None else f"advance to {successor['name']}"
    return {"pipeline": state["pipeline_name"], "status": state["status"], "current_phase": current, "phases": phases, "next_action": action}


__all__ = [
    "PipelineError", "atomic_write_json", "collect_results", "effective_selection",
    "estimated_phase_counts", "expand_phase", "initial_state", "load_manifest",
    "load_state", "materialize_phase", "next_phase", "parent_records", "phase_by_name",
    "pipeline_hash", "pipeline_paths", "plan_phase", "rank_results", "resolve_pipeline_path",
    "retry_decision", "scientific_config_hash", "state_lock", "status_summary",
    "update_phase_ranking", "validate_generated_records", "write_state",
    "validate_pipeline_semantics",
]
