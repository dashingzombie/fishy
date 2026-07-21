from __future__ import annotations

import copy
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from fish_species.sweeps.core import (
    PipelineError,
    atomic_write_json,
    collect_local_results,
    expand_phase,
    initial_state,
    load_manifest,
    load_state,
    materialize_phase,
    pipeline_paths,
    rank_results,
    retry_decision,
    scientific_config_hash,
    write_state,
)
from fish_species.sweeps.pipeline import plan_command, resume_command, submit_command
from fish_species.sweeps.schema import PipelineConfigError, validate_pipeline
from fish_species.sweeps.execution import submit_auto_advance, submit_phase_slurm
from fish_species.slurm.submission import RecordingSbatchClient


REPO = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO / "configs" / "experiments" / "long_tail.yaml"


def make_document(root: Path, phases: list[dict] | None = None) -> dict:
    document = {
        "pipeline": {
            "name": "synthetic",
            "base_config": str(BASE_CONFIG),
            "output_root": str(root / "outputs"),
            "state_file": str(root / "outputs" / "state.json"),
            "execution": {
                "backend": "slurm", "auto_advance": False,
                "retry_failed": True, "maximum_retries": 1,
                "array_max_active": 2,
            },
            "results": {
                "source": "local", "metric_file": "metrics/validation_summary.json",
                "require_status": "completed",
            },
            "selection": {
                "primary_metric": "species_balanced_accuracy", "direction": "max",
                "top_k": 1, "tie_tolerance": 1e-6,
                "tie_breakers": [{"metric": "species_macro_f1", "direction": "max"}],
                "constraints": [],
            },
            "phases": phases or [{
                "name": "first", "type": "cartesian", "parent": "base",
                "parameters": {"training.lr": [1e-5, 3e-5]},
            }],
        },
        "_pipeline_file": str(root / "pipeline.yaml"),
    }
    validate_pipeline({"pipeline": copy.deepcopy(document["pipeline"])})
    return document


class PipelineSchemaTests(unittest.TestCase):
    def test_schema_dependency_and_cycle_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            bad = copy.deepcopy(document)
            bad.pop("_pipeline_file")
            bad["pipeline"]["execution"]["backend"] = "kubernetes"
            with self.assertRaises(PipelineConfigError):
                validate_pipeline(bad)

            cycle = copy.deepcopy(document)
            cycle.pop("_pipeline_file")
            cycle["pipeline"]["phases"] = [
                {"name": "a", "type": "conditions", "parent": "b", "conditions": [{"name": "x", "overrides": {}}]},
                {"name": "b", "type": "conditions", "parent": "a", "conditions": [{"name": "x", "overrides": {}}]},
            ]
            with self.assertRaisesRegex(PipelineConfigError, "cycle"):
                validate_pipeline(cycle)

    def test_cartesian_and_atomic_condition_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            phase = document["pipeline"]["phases"][0]
            phase["parameters"]["model.name"] = ["convnext_base", "vit_b_16"]
            records = expand_phase(document, phase)
            self.assertEqual(len(records), 4)
            self.assertEqual(len({row["configuration_hash"] for row in records}), 4)

            phase = {
                "name": "atomic", "type": "conditions", "parent": "base",
                "conditions": [
                    {"name": "a", "overrides": {"training.lr": 1e-5, "training.epochs": 2}},
                    {"name": "b", "overrides": {"training.lr": 2e-5, "training.epochs": 3}},
                ],
            }
            records = expand_phase(document, phase)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["resolved_config"]["training"]["epochs"], 2)

    def test_invalid_phase_type_and_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            raw = {"pipeline": copy.deepcopy(document["pipeline"])}
            raw["pipeline"]["phases"][0]["type"] = "mega_product"
            with self.assertRaises(PipelineConfigError):
                validate_pipeline(raw)
            raw = {"pipeline": copy.deepcopy(document["pipeline"])}
            raw["pipeline"]["phases"][0]["parent"] = "absent"
            with self.assertRaisesRegex(PipelineConfigError, "unknown parent"):
                validate_pipeline(raw)


class HashAndInheritanceTests(unittest.TestCase):
    def test_hash_is_deterministic_and_excludes_ephemeral_fields(self):
        config = {"model": {"name": "x"}, "training": {"lr": 1e-5}}
        changed = copy.deepcopy(config)
        changed.update({
            "output": {"out_dir": "/different"},
            "slurm": {"job_id": "999"},
            "pipeline_run": {"run_id": "ephemeral"},
        })
        self.assertEqual(scientific_config_hash(config), scientific_config_hash(changed))
        changed["training"]["lr"] = 2e-5
        self.assertNotEqual(scientific_config_hash(config), scientific_config_hash(changed))

    def test_parent_inheritance_evaluation_and_fine_tune(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            parent_config = yaml.safe_load(BASE_CONFIG.read_text())
            # Load through expansion first so the full extends tree is represented.
            base_parent = expand_phase(document, document["pipeline"]["phases"][0])[0]
            parent_config = base_parent["resolved_config"]
            parent_config["training"]["lr"] = 0.123
            parent = {
                "run_id": "parent", "configuration_hash": scientific_config_hash(parent_config),
                "checkpoint": str(Path(directory) / "parent.pt"),
                "metrics": {"species_balanced_accuracy": 0.5},
                "resolved_config": parent_config,
            }
            evaluation = {
                "name": "evaluate", "type": "evaluation_only", "parent": "first",
                "conditions": [{"name": "risk", "overrides": {"inference.taxonomic_minimum_risk.enabled": True}}],
            }
            child = expand_phase(document, evaluation, [parent])[0]
            self.assertEqual(child["resolved_config"]["training"]["lr"], 0.123)
            self.assertEqual(child["resolved_config"]["pipeline_run"]["parent_run_id"], "parent")
            self.assertEqual(child["resolved_config"]["pipeline_run"]["execution_mode"], "evaluation_only")
            self.assertTrue(child["resolved_config"]["fine_tuning"]["reset_optimizer"])

            fine_tune = {
                "name": "ft", "type": "fine_tune", "parent": "evaluate",
                "overrides": {"preprocessing.image_size": 320},
                "checkpoint": {"source": "parent_best", "load_model": True, "load_optimizer": False},
            }
            result = expand_phase(document, fine_tune, [parent])[0]
            self.assertEqual(result["resolved_config"]["preprocessing"]["image_size"], 320)
            self.assertEqual(result["resolved_config"]["pipeline_run"]["execution_mode"], "train")

    def test_scientific_notation_is_preserved_in_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            records = expand_phase(document, document["pipeline"]["phases"][0])
            materialize_phase(document, document["pipeline"]["phases"][0], records)
            loaded = yaml.safe_load(Path(records[0]["resolved_config_path"]).read_text())
            self.assertIsInstance(loaded["training"]["lr"], float)
            self.assertAlmostEqual(loaded["training"]["lr"], 1e-5)


class PersistenceAndDryRunTests(unittest.TestCase):
    def test_manifest_generation_is_idempotent_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            phase = document["pipeline"]["phases"][0]
            records = expand_phase(document, phase)
            path = materialize_phase(document, phase, records)
            before = path.read_bytes()
            materialize_phase(document, phase, records)
            self.assertEqual(before, path.read_bytes())
            manifest = load_manifest(document, "first")
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(manifest["runs"][0]["slurm_array_index"], 0)

    def test_state_atomic_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            state = initial_state(document)
            write_state(document, state)
            loaded = load_state(document)
            self.assertEqual(loaded["pipeline_name"], "synthetic")
            _, state_path = pipeline_paths(document)
            self.assertFalse(list(state_path.parent.glob(".state.json.tmp-*")))

    def test_plan_and_submission_dry_run_never_call_subprocess_or_write_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            with mock.patch.object(subprocess, "run") as run:
                report = plan_command(document)
                submit = submit_command(document, None, dry_run=True)
            run.assert_not_called()
            self.assertEqual(report["first_phase_runs"], 2)
            self.assertTrue(submit["dry_run"])
            self.assertFalse(pipeline_paths(document)[1].exists())
            self.assertFalse((root / "outputs").exists())

    def test_duplicate_submission_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            state = initial_state(document)
            phase = document["pipeline"]["phases"][0]
            records = expand_phase(document, phase)
            materialize_phase(document, phase, records)
            phase_state = state["phases"]["first"]
            phase_state.update({"status": "submitted", "planned_runs": 2, "runs": {
                row["run_id"]: {"completion_status": "pending", "submission_status": "submitted", "retry_count": 0}
                for row in records
            }})
            write_state(document, state)
            with mock.patch.object(subprocess, "run") as run:
                with self.assertRaisesRegex(PipelineError, "already submitted"):
                    submit_command(document, None)
            run.assert_not_called()

    def test_slurm_submission_uses_injectable_mock_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            phase = document["pipeline"]["phases"][0]
            records = expand_phase(document, phase)
            materialize_phase(document, phase, records)
            client = RecordingSbatchClient(["123456"])
            submitted, manifest = submit_phase_slurm(
                document, "first", records,
                str(REPO / "configs" / "clusters" / "local.yaml"),
                1, client=client,
            )
            self.assertEqual(submitted["train_array"], "123456")
            self.assertTrue(manifest.is_file())
            self.assertTrue(client.calls)
            self.assertEqual(client.calls[0][0:2], ["sbatch", "--parsable"])

    def test_auto_advance_uses_afterany_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            document = make_document(Path(directory))
            client = RecordingSbatchClient(["234567"])
            job_id = submit_auto_advance(
                document, "123456",
                str(REPO / "configs" / "clusters" / "local.yaml"),
                client=client,
            )
            self.assertEqual(job_id, "234567")
            self.assertIn("--dependency=afterany:123456", client.calls[0])
            self.assertTrue(any(value.startswith("--wrap=") for value in client.calls[0]))


class ResultAndRankingTests(unittest.TestCase):
    def _prepared(self, root: Path, *, constraints: list[dict] | None = None):
        document = make_document(root)
        if constraints is not None:
            document["pipeline"]["selection"]["constraints"] = constraints
        phase = document["pipeline"]["phases"][0]
        records = expand_phase(document, phase)
        materialize_phase(document, phase, records)
        state = initial_state(document)
        state["phases"]["first"].update({
            "status": "submitted", "planned_runs": len(records),
            "runs": {row["run_id"]: {
                "completion_status": "pending", "submission_status": "submitted",
                "retry_count": 0, "metrics": {}, "checkpoint": None,
            } for row in records},
        })
        return document, phase, records, state

    @staticmethod
    def _write_result(record: dict, metrics: dict, *, digest: str | None = None, status: str = "completed"):
        output = Path(record["expected_output_directory"])
        (output / "metrics").mkdir(parents=True)
        checkpoint = output / "best_model.pt"
        checkpoint.write_bytes(b"checkpoint")
        (output / "metrics" / "validation_summary.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (output / "run_status.json").write_text(json.dumps({
            "status": status, "exit_code": 0 if status == "completed" else 1,
            "configuration_hash": digest or record["configuration_hash"],
            "best_checkpoint": str(checkpoint), "best_epoch": 3,
        }), encoding="utf-8")

    def test_result_matching_and_invalid_metric_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            document, phase, records, state = self._prepared(Path(directory))
            self._write_result(records[0], {"species_balanced_accuracy": 0.7, "species_macro_f1": 0.6})
            self._write_result(records[1], {"species_balanced_accuracy": float("nan")})
            rows = collect_local_results(document, state, phase)
            self.assertTrue(rows[0]["valid_result"])
            self.assertFalse(rows[1]["valid_result"])
            self.assertEqual(rows[1]["failure_category"], "invalid_metrics")

    def test_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            document, phase, records, state = self._prepared(Path(directory))
            self._write_result(records[0], {"species_balanced_accuracy": 0.5}, digest="wrong")
            rows = collect_local_results(document, state, phase)
            self.assertEqual(rows[0]["failure_category"], "configuration_hash_mismatch")

    def test_coverage_constraint_top_k_and_deterministic_tie_break(self):
        constraint = [{
            "metric": "species_represented_species_count", "operator": "greater_equal",
            "value_from": "phase_max", "fraction": 0.95,
        }]
        with tempfile.TemporaryDirectory() as directory:
            document, phase, records, state = self._prepared(Path(directory), constraints=constraint)
            state_selection = document["pipeline"]["selection"]
            state_selection["top_k"] = 1
            rows = []
            for index, record in enumerate(records):
                rows.append({
                    **record, "completion_status": "successful", "valid_result": True,
                    "metrics": {
                        "species_balanced_accuracy": 0.8,
                        "species_macro_f1": 0.7,
                        "species_represented_species_count": 100 if index == 0 else 90,
                    }, "rejection_reason": None,
                })
            ranked = rank_results(document, phase, rows)
            selected = [row for row in ranked if row["selected_rank"]]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["metrics"]["species_represented_species_count"], 100)
            self.assertEqual(next(row for row in ranked if row["metrics"]["species_represented_species_count"] == 90)["constraint_status"], "failed")

    def test_remaining_tie_break_uses_configuration_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            document, phase, records, _ = self._prepared(Path(directory))
            rows = [{
                **record, "completion_status": "successful", "valid_result": True,
                "metrics": {"species_balanced_accuracy": 0.8, "species_macro_f1": 0.7},
                "rejection_reason": None,
            } for record in records]
            ranked = rank_results(document, phase, rows)
            self.assertEqual(
                ranked[0]["configuration_hash"],
                min(record["configuration_hash"] for record in records),
            )

    def test_baseline_relative_constraint_is_per_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = make_document(root)
            document["pipeline"]["selection"]["constraints"] = [{
                "metric": "species_head_over_10_accuracy",
                "operator": "drop_from_phase_baseline_less_equal", "value": 0.02,
            }]
            phase = document["pipeline"]["phases"][0]
            records = expand_phase(document, phase)
            materialize_phase(document, phase, records)
            rows = []
            for index, record in enumerate(records):
                rows.append({
                    **record, "baseline": index == 0, "completion_status": "successful",
                    "valid_result": True, "rejection_reason": None,
                    "metrics": {
                        "species_balanced_accuracy": 0.5 + index * 0.1,
                        "species_macro_f1": 0.5,
                        "species_head_over_10_accuracy": 0.8 if index == 0 else 0.77,
                    },
                })
            ranked = rank_results(document, phase, rows)
            candidate = next(row for row in ranked if not row["baseline"])
            self.assertFalse(candidate["valid_result"])
            self.assertIn("constraint failed", candidate["rejection_reason"])

    def test_retry_policy(self):
        execution = {"retry_failed": True, "maximum_retries": 1}
        self.assertTrue(retry_decision({"failure_category": "preempted", "retry_count": 0}, execution)[0])
        self.assertFalse(retry_decision({"failure_category": "cuda_oom", "retry_count": 0}, execution)[0])
        self.assertFalse(retry_decision({"failure_category": "infrastructure", "retry_count": 1}, execution)[0])

    def test_resume_dry_run_preserves_success_and_identifies_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            document, phase, records, state = self._prepared(Path(directory))
            self._write_result(records[0], {"species_balanced_accuracy": 0.7, "species_macro_f1": 0.6})
            write_state(document, state)
            with mock.patch.object(subprocess, "run") as run:
                report = resume_command(document, None, dry_run=True)
            run.assert_not_called()
            self.assertEqual(report["successful_preserved"], [records[0]["run_id"]])
            self.assertEqual(report["incomplete"], [records[1]["run_id"]])


if __name__ == "__main__":
    unittest.main()
