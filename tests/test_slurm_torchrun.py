from __future__ import annotations

from pathlib import Path
import unittest

from fish_species.slurm.planning import SubmissionPlan
from fish_species.slurm.rendering import _array_context


class SlurmTorchrunTests(unittest.TestCase):
    def test_multi_gpu_array_uses_one_node_torchrun(self):
        plan = SubmissionPlan(
            schema_version=2,
            experiment_type="standard",
            cluster_profile="genome",
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


if __name__ == "__main__":
    unittest.main()

