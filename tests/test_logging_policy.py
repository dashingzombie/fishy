from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fish_species.config import load_config, validate_config
from fish_species.config.validation import ConfigValidationError
from fish_species.logging.wandb_logger import WandbLogger
from fish_species.evaluation.condition_matrix import _write_task_reports


class _Artifact:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class _Backend:
    Artifact = _Artifact


class _Run:
    def __init__(self):
        self.artifacts: list[_Artifact] = []

    def log_artifact(self, artifact: _Artifact) -> None:
        self.artifacts.append(artifact)


class LoggingPolicyTests(unittest.TestCase):
    def test_wandb_artifacts_never_include_model_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            checkpoint = root / "best_model.pt"
            metrics.write_text("{}", encoding="utf-8")
            checkpoint.write_bytes(b"model")
            run = _Run()
            logger = WandbLogger(
                cfg={"wandb": {"enabled": True}}, run_name="test",
                out_dir=root, backend=_Backend(), run=run,
            )
            logged = logger.log_artifacts([metrics, checkpoint])
            self.assertEqual(logged, ["test-scientific-record"])
            self.assertEqual(len(run.artifacts), 1)
            self.assertEqual(run.artifacts[0].files, [str(metrics)])
            self.assertFalse(logger.should_log_confusion_matrix("original", "species"))

    def test_configuration_rejects_wandb_model_upload(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs" / "experiments" / "long_tail.yaml")
        config["wandb"]["log_model"] = True
        with self.assertRaises(ConfigValidationError):
            validate_config(
                config, workflow="training", check_paths=False,
                check_model_registry=False,
            )

    def test_evaluation_reports_do_not_create_confusion_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_task_reports(
                root, "original", {"species": [0, 1]}, {"species": [0, 0]},
                {"species": {0: "a", 1: "b"}},
            )
            self.assertTrue(
                (root / "classification_reports" / "original" / "classification_report_species.csv").is_file()
            )
            self.assertFalse((root / "confusion_matrices").exists())


if __name__ == "__main__":
    unittest.main()
