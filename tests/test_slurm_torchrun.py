from __future__ import annotations

from pathlib import Path
import unittest

from fish_species.slurm.planning import SubmissionPlan
from fish_species.slurm.rendering import _array_context
from fish_species.slurm.rendering import _select_array_template, render_template


class SlurmTorchrunTests(unittest.TestCase):
    def _plan(self, cluster: str = "genome") -> SubmissionPlan:
        return SubmissionPlan(
            schema_version=2,
            experiment_type="standard",
            cluster_profile=cluster,
            results_root="outputs",
            array_size=1,
            array_max_active=1,
            models=("dinov3_vits16",),
            conditions=("original",),
            training_modes=("multitask",),
            run_specs=(),
            dependencies=(),
            canonical_trainer_command=(
                "python", "-m", "fish_species.training", "--config",
                "resolved_run_config.yaml", "--single-run",
            ),
            resolved_config_sha256="0" * 64,
        )

    def test_multi_gpu_array_uses_one_node_torchrun(self):
        plan = self._plan()
        config = {
            "training": {"distributed": {"enabled": True}},
            "slurm": {
                "gpus_per_task": 2,
                "paths": {"project_root": ".", "data_root": "data"},
                "scratch": {"root": "/tmp/fish-test"},
                "environment": {},
            },
        }
        context = _array_context(plan, config, Path("slurm/generated/test"), node_local=False)
        command = context["TRAIN_COMMAND"]
        self.assertIn("torchrun", command)
        self.assertIn("--nnodes=1", command)
        self.assertIn("--nproc_per_node=2", command)

    def test_ghpc_uses_shared_inputs_and_persistent_node_cache(self):
        config = {
            "training": {"distributed": {"enabled": False}},
            "slurm": {
                "gpus_per_task": 1,
                "planning": {"external_expansion": "sweep"},
                "paths": {
                    "project_root": "/shared/project",
                    "data_root": "/shared/data",
                    "metadata_csv": "/shared/data/labels.json",
                    "cache_root": "/scratch/user/fish_species_image_cache",
                },
                "scratch": {
                    "mode": "node_local_cache",
                    "root": "/scratch/user/run_20260721_120000_1",
                    "copy_project": False,
                    "copy_data": False,
                },
                "environment": {},
            },
        }
        self.assertEqual(_select_array_template(config), "node_local_array_job.sh.tmpl")
        context = _array_context(
            self._plan("ghpc"), config, Path("slurm/generated/test"),
            node_local=True,
        )
        self.assertEqual(context["PROJECT_ROOT"], "/shared/project")
        self.assertEqual(context["DATA_ROOT"], "/shared/data")
        self.assertEqual(context["CACHE_ROOT"], "/scratch/user/fish_species_image_cache")
        self.assertEqual(
            context["CACHE_READY_MARKER"],
            "/scratch/user/fish_species_image_cache/CACHE_READY",
        )

    def test_genome_template_copies_and_explicitly_cleans_staged_cache(self):
        config = {
            "training": {"distributed": {"enabled": False}},
            "slurm": {
                "gpus_per_task": 1,
                "paths": {
                    "project_root": "/shared/project", "data_root": "/shared/data",
                    "metadata_csv": "/shared/data/labels.json",
                    "cache_root": "/shared/cache",
                },
                "scratch": {
                    "mode": "job_local_cache", "root": "/tmp/user/fish",
                    "copy_cache_to_tmp": 1, "cleanup_after_run": True,
                },
                "environment": {},
            },
        }
        context = _array_context(
            self._plan(), config, Path("slurm/generated/test"), node_local=False
        )
        rendered = render_template("persistent_cache_array_job.sh.tmpl", context)
        self.assertIn('rsync -a "$CACHE_ROOT/" "$LOCAL_CACHE/"', rendered)
        self.assertIn('rm -rf -- "$LOCAL_CACHE"', rendered)
        self.assertIn("CLEANUP_AFTER_RUN=true", rendered)


if __name__ == "__main__":
    unittest.main()
