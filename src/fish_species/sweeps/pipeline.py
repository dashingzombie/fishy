"""Command-line interface for deterministic sequential sweep pipelines."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .core import (
    PipelineError, collect_results, effective_selection, estimated_phase_counts,
    expand_phase, initial_state, load_manifest, load_state, next_phase,
    phase_by_name, pipeline_paths, plan_phase, rank_results, state_lock,
    status_summary, update_phase_ranking, validate_generated_records, write_state,
    validate_pipeline_semantics,
)
from .execution import (
    active_slurm_jobs, cancel_jobs, execute_local_records, render_phase_bundle, submit_auto_advance,
    submit_phase_slurm,
)
from .schema import PipelineConfigError, load_pipeline


def _print(value: object, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
    elif isinstance(value, str):
        print(value)
    else:
        print(yaml.safe_dump(value, sort_keys=False).rstrip())


def plan_command(document: Mapping[str, Any], *, artifacts_dir: str | None = None) -> dict[str, Any]:
    """Validate and report the pipeline without touching execution state."""
    first = document["pipeline"]["phases"][0]
    records = expand_phase(document, first)
    semantic_counts = validate_pipeline_semantics(document)
    counts = estimated_phase_counts(document)
    selection = effective_selection(document, first)
    report = {
        "pipeline": document["pipeline"]["name"],
        "first_phase": first["name"], "first_phase_runs": len(records),
        "phase_counts": counts,
        "validated_phase_run_counts": semantic_counts,
        "ranking": {
            "primary_metric": selection.get("primary_metric"),
            "direction": selection.get("direction"),
            "top_k": selection.get("top_k"),
            "tie_breakers": selection.get("tie_breakers", []),
            "constraints": selection.get("constraints", []),
        },
        "configuration_hashes": [record["configuration_hash"] for record in records],
        "planned_commands": [
            [sys.executable, "-m", "fish_species.training", "--config", path, "--single-run"]
            for path in (
                str(Path(artifacts_dir).resolve() / "configs" / f"{record['run_id']}.yaml")
                if artifacts_dir else f"<not-written>/{record['run_id']}.yaml"
                for record in records
            )
        ],
        "generated_configuration_paths": [
            str(Path(artifacts_dir).resolve() / "configs" / f"{record['run_id']}.yaml")
            if artifacts_dir else f"<not-written>/{record['run_id']}.yaml"
            for record in records
        ],
    }
    if artifacts_dir:
        root = Path(artifacts_dir).resolve()
        if root.exists():
            raise PipelineError(f"planning artifacts directory already exists: {root}")
        configs = root / "configs"
        configs.mkdir(parents=True)
        for record in records:
            (configs / f"{record['run_id']}.yaml").write_text(
                yaml.safe_dump(record["resolved_config"], sort_keys=False), encoding="utf-8"
            )
        (root / "plan.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def _records_for_submission(
    document: Mapping[str, Any], state: Mapping[str, Any], phase_name: str,
    *, force: bool,
) -> list[dict[str, Any]]:
    manifest = load_manifest(document, phase_name)
    dynamic = state["phases"][phase_name]["runs"]
    execution = document["pipeline"].get("execution", {}) or {}
    records = []
    for item in manifest["runs"]:
        run = dynamic[item["run_id"]]
        status = run.get("completion_status", "pending")
        if status == "successful":
            continue
        if status in {"failed", "missing"}:
            from .core import retry_decision
            eligible, _ = retry_decision(run, execution)
            if not eligible and not force:
                continue
        record = copy.deepcopy(item)
        record["resolved_config"] = None
        records.append(record)
    return records


def _submit_current(
    document: Mapping[str, Any], state: dict[str, Any], phase: Mapping[str, Any],
    cluster_config: str | None, *, force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    phase_state = state["phases"][phase["name"]]
    if phase_state["status"] == "not_planned":
        plan_phase(document, state, phase)
    elif phase_state["status"] in {"submitted", "running"} and not force:
        raise PipelineError(
            f"phase {phase['name']!r} is already submitted; pass --force only to recover a known scheduler issue"
        )
    records = _records_for_submission(document, state, str(phase["name"]), force=force)
    if not records:
        raise PipelineError(f"phase {phase['name']!r} has no eligible incomplete runs")
    attempt = int(phase_state.get("attempts", 0)) + 1
    backend = str(document["pipeline"].get("execution", {}).get("backend", "slurm"))
    if dry_run:
        if backend == "local":
            commands = [
                [sys.executable, "-m", "fish_species.training", "--config", item["resolved_config_path"], "--single-run"]
                for item in records
            ]
        else:
            commands = [["sbatch", "--parsable", f"<phase:{phase['name']}:attempt:{attempt}>"]]
        return {"phase": phase["name"], "runs": len(records), "backend": backend, "commands": commands, "dry_run": True}

    for item in records:
        dynamic = phase_state["runs"][item["run_id"]]
        if dynamic.get("completion_status") in {"failed", "missing"}:
            dynamic["retry_count"] = int(dynamic.get("retry_count", 0)) + 1
        dynamic.update({"submission_status": "submitting", "completion_status": "pending"})
    phase_state["attempts"] = attempt
    phase_state["status"] = "submitting"
    write_state(document, state)

    if backend == "local":
        phase_state["status"] = "running"
        for item in records:
            phase_state["runs"][item["run_id"]]["submission_status"] = "local"
        write_state(document, state)
        returncodes = execute_local_records(records)
        return {"phase": phase["name"], "backend": "local", "returncodes": returncodes}

    submitted, manifest_path = submit_phase_slurm(
        document, str(phase["name"]), records, cluster_config, attempt,
    )
    train_job_id = submitted.get("train_array")
    if train_job_id is None:
        raise PipelineError(f"SLURM submission did not return train_array: {submitted}")
    phase_state["slurm_job_ids"].append(train_job_id)
    phase_state["status"] = "submitted"
    for item in records:
        phase_state["runs"][item["run_id"]].update({
            "submission_status": "submitted", "slurm_job_id": train_job_id,
        })
    # Persist the array ID before the optional collector submission. If the
    # latter fails, resume still knows that the expensive array already exists.
    write_state(document, state)
    collector = None
    if bool(document["pipeline"].get("execution", {}).get("auto_advance", False)):
        collector = submit_auto_advance(document, train_job_id, cluster_config)
        phase_state["collector_job_ids"].append(collector)
    return {
        "phase": phase["name"], "backend": "slurm", "runs": len(records),
        "job_ids": submitted, "collector_job_id": collector,
        "artifact_manifest": str(manifest_path),
    }


def submit_command(
    document: Mapping[str, Any], cluster_config: str | None,
    *, force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Submit the first incomplete phase without duplicating successful runs."""
    _, state_path = pipeline_paths(document)
    if dry_run:
        state = load_state(document, required=False) or initial_state(document)
        phase = next(
            (
                item for item in document["pipeline"]["phases"]
                if state["phases"][item["name"]]["status"] != "completed"
            ),
            None,
        )
        if phase is None:
            return {"dry_run": True, "action": "pipeline already complete", "commands": []}
        if state["phases"][phase["name"]]["status"] == "not_planned":
            records = expand_phase(document, phase)
            validate_generated_records(records)
            return {"phase": phase["name"], "runs": len(records), "backend": document["pipeline"].get("execution", {}).get("backend", "slurm"), "commands": [["sbatch", "--parsable", "<rendered-array>"]], "dry_run": True}
        return _submit_current(document, state, phase, cluster_config, force=force, dry_run=True)
    with state_lock(state_path):
        state = load_state(document, required=False) or initial_state(document)
        phase = next(
            (
                item for item in document["pipeline"]["phases"]
                if state["phases"][item["name"]]["status"] != "completed"
            ),
            None,
        )
        if phase is None:
            raise PipelineError("pipeline is already complete")
        state["current_phase"] = phase["name"]
        result = _submit_current(document, state, phase, cluster_config, force=force)
        state["status"] = "running"
        write_state(document, state)
        return result


def _manual_parents(state: dict[str, Any], phase: Mapping[str, Any], parents: Sequence[str]) -> None:
    parent_name = str(phase.get("parent", "base"))
    if parent_name == "base":
        raise PipelineError("the first phase has no selectable parents")
    known = state["phases"][parent_name]["runs"]
    unknown = [run_id for run_id in parents if run_id not in known]
    if unknown:
        raise PipelineError(f"unknown parent run IDs: {', '.join(unknown)}")
    unsuccessful = [run_id for run_id in parents if known[run_id].get("completion_status") != "successful"]
    if unsuccessful:
        raise PipelineError(f"manual parents are not successful: {', '.join(unsuccessful)}")
    state["phases"][parent_name]["selected_run_ids"] = list(parents)


def advance_command(
    document: Mapping[str, Any], cluster_config: str | None,
    *, submit: bool = False, parents: Sequence[str] = (), force: bool = False,
) -> dict[str, Any]:
    """Collect/rank the current phase and generate its successor."""
    _, state_path = pipeline_paths(document)
    if not submit:
        state = copy.deepcopy(load_state(document))
        current = phase_by_name(document, str(state["current_phase"]))
        rows = collect_results(document, state, current)
        ranked = rank_results(document, current, rows)
        selected = [row["run_id"] for row in ranked if row.get("selected_rank")]
        successor = next_phase(document, str(current["name"]))
        report: dict[str, Any] = {
            "dry_run": True, "collected_phase": current["name"],
            "selected_parents": selected,
            "valid_runs": sum(bool(row.get("valid_result")) for row in ranked),
        }
        if successor and selected:
            state["phases"][current["name"]]["selected_run_ids"] = selected
            if parents:
                _manual_parents(state, successor, parents)
            generated = expand_phase(document, successor, __import__("fish_species.sweeps.core", fromlist=["parent_records"]).parent_records(document, state, successor))
            validate_generated_records(generated)
            report.update({"next_phase": successor["name"], "next_phase_runs": len(generated), "configuration_hashes": [item["configuration_hash"] for item in generated]})
            backend = document["pipeline"].get("execution", {}).get("backend", "slurm")
            report["planned_commands"] = (
                [[sys.executable, "-m", "fish_species.training", "--config", f"<immutable-config:{item['run_id']}>", "--single-run"] for item in generated]
                if backend == "local" else [["sbatch", "--parsable", f"<phase:{successor['name']}:array-size:{len(generated)}>"]]
            )
        else:
            report["next_phase"] = None
        return report

    with state_lock(state_path):
        state = load_state(document)
        current = phase_by_name(document, str(state["current_phase"]))
        phase_state = state["phases"][current["name"]]
        if (
            document["pipeline"].get("execution", {}).get("backend", "slurm") == "slurm"
            and phase_state.get("status") in {"submitted", "running"}
            and not force
        ):
            active = active_slurm_jobs(phase_state.get("slurm_job_ids", []))
            if active:
                raise PipelineError(
                    "training jobs are still active; refusing early collection/resubmission: "
                    + ", ".join(active)
                )
        rows = collect_results(document, state, current)
        ranked = update_phase_ranking(document, state, current, rows)
        eligible_retries = []
        from .core import retry_decision
        for row in ranked:
            dynamic = state["phases"][current["name"]]["runs"][row["run_id"]]
            if dynamic.get("completion_status") in {"failed", "missing"} and retry_decision(dynamic, document["pipeline"].get("execution", {}))[0]:
                eligible_retries.append(row["run_id"])
        if eligible_retries:
            state["phases"][current["name"]]["status"] = "partial"
            write_state(document, state)
            result = _submit_current(document, state, current, cluster_config, force=False)
            write_state(document, state)
            return {"action": "retry", "run_ids": eligible_retries, **result}
        selected = state["phases"][current["name"]]["selected_run_ids"]
        if not selected:
            state["status"] = "failed"
            write_state(document, state)
            raise PipelineError(f"phase {current['name']!r} produced no valid selected runs")
        successor = next_phase(document, str(current["name"]))
        if successor is None:
            state.update({"status": "completed", "current_phase": current["name"]})
            write_state(document, state)
            return {"action": "complete", "phase": current["name"], "selected": selected}
        if parents:
            _manual_parents(state, successor, parents)
        state["current_phase"] = successor["name"]
        records = plan_phase(document, state, successor)
        write_state(document, state)
        result = _submit_current(document, state, successor, cluster_config, force=force)
        state["status"] = "running"
        write_state(document, state)
        return {"action": "advance", "selected_parents": selected, "next_phase_runs": len(records), **result}


def resume_command(
    document: Mapping[str, Any], cluster_config: str | None, *, dry_run: bool = False,
) -> dict[str, Any]:
    """Resume from persisted state while preserving every successful run."""
    state = load_state(document, required=False)
    if state is None:
        return submit_command(document, cluster_config, dry_run=dry_run)
    current = phase_by_name(document, str(state["current_phase"]))
    if state["phases"][current["name"]]["status"] == "not_planned":
        return submit_command(document, cluster_config, dry_run=dry_run)
    if dry_run:
        copied = copy.deepcopy(state)
        rows = collect_results(document, copied, current)
        return {
            "dry_run": True, "phase": current["name"],
            "successful_preserved": [row["run_id"] for row in rows if row.get("completion_status") == "successful"],
            "incomplete": [row["run_id"] for row in rows if row.get("completion_status") != "successful"],
        }
    return advance_command(document, cluster_config, submit=True)


def status_command(document: Mapping[str, Any]) -> dict[str, Any]:
    """Read state and derive current filesystem status without mutating it."""
    state = load_state(document, required=False)
    if state is not None:
        copied = copy.deepcopy(state)
        current = phase_by_name(document, str(copied["current_phase"]))
        if copied["phases"][current["name"]]["status"] != "not_planned":
            try:
                rows = collect_results(document, copied, current)
                if document["pipeline"].get("execution", {}).get("backend", "slurm") == "slurm":
                    active = active_slurm_jobs(
                        copied["phases"][current["name"]].get("slurm_job_ids", [])
                    )
                    if active:
                        for row in rows:
                            if row.get("completion_status") == "missing":
                                copied["phases"][current["name"]]["runs"][row["run_id"]]["completion_status"] = "running"
                        values = copied["phases"][current["name"]]["runs"].values()
                        copied["phases"][current["name"]]["running_runs"] = sum(item.get("completion_status") == "running" for item in values)
                        copied["phases"][current["name"]]["missing_runs"] = sum(item.get("completion_status") == "missing" for item in values)
            except PipelineError:
                pass
        return status_summary(document, copied)
    return status_summary(document, None)


def cancel_command(document: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Cancel only active job IDs recorded by this pipeline."""
    _, state_path = pipeline_paths(document)
    if dry_run:
        state = load_state(document)
        jobs = []
        for phase in state["phases"].values():
            if phase["status"] in {"submitted", "running", "submitting"}:
                jobs.extend(phase.get("slurm_job_ids", []))
                jobs.extend(phase.get("collector_job_ids", []))
        return {"dry_run": True, "commands": cancel_jobs(jobs, dry_run=True)}
    with state_lock(state_path):
        state = load_state(document)
        jobs = []
        for phase in state["phases"].values():
            if phase["status"] in {"submitted", "running", "submitting"}:
                jobs.extend(phase.get("slurm_job_ids", []))
                jobs.extend(phase.get("collector_job_ids", []))
        commands = cancel_jobs(jobs)
        state["status"] = "cancelled"
        for phase in state["phases"].values():
            if phase["status"] in {"submitted", "running", "submitting"}:
                phase["status"] = "cancelled"
        write_state(document, state)
        return {"cancelled_job_ids": [command[-1] for command in commands], "state_retained": True}


def build_parser() -> argparse.ArgumentParser:
    """Build the public pipeline CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "submit", "status", "advance", "resume", "cancel"):
        command = subparsers.add_parser(name)
        command.add_argument("--pipeline", required=True)
        command.add_argument("--json", action="store_true")
        if name in {"submit", "advance", "resume"}:
            command.add_argument("--cluster-config")
        if name == "plan":
            command.add_argument("--artifacts-dir")
        if name == "submit":
            command.add_argument("--force", action="store_true")
            command.add_argument("--dry-run", action="store_true")
        if name == "advance":
            command.add_argument("--submit", action="store_true")
            command.add_argument("--force", action="store_true")
            command.add_argument("--parents", help="comma-separated successful parent run IDs")
        if name == "resume":
            command.add_argument("--dry-run", action="store_true")
        if name == "cancel":
            command.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one pipeline operation."""
    args = build_parser().parse_args(argv)
    try:
        document = load_pipeline(args.pipeline)
        if args.command == "plan":
            result = plan_command(document, artifacts_dir=args.artifacts_dir)
        elif args.command == "submit":
            result = submit_command(document, args.cluster_config, force=args.force, dry_run=args.dry_run)
        elif args.command == "status":
            result = status_command(document)
        elif args.command == "advance":
            parents = tuple(item for item in (args.parents or "").split(",") if item)
            result = advance_command(document, args.cluster_config, submit=args.submit, parents=parents, force=args.force)
        elif args.command == "resume":
            result = resume_command(document, args.cluster_config, dry_run=args.dry_run)
        else:
            result = cancel_command(document, dry_run=args.dry_run)
        _print(result, as_json=args.json)
        return 0
    except (PipelineError, PipelineConfigError, ValueError) as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "advance_command", "build_parser", "cancel_command", "main", "plan_command",
    "resume_command", "status_command", "submit_command",
]
