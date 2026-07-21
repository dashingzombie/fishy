from __future__ import annotations

import tempfile
import unittest
import copy
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from fish_species.config.loading import load_config
from fish_species.config.validation import ConfigValidationError, validate_config
from fish_species.evaluation.taxonomic import (
    minimum_taxonomic_risk_predictions,
    taxonomic_costs,
    taxonomic_metrics,
)
from fish_species.models.long_tail import CosineClassifier, PrototypeClassifier
from fish_species.models.multitask import MultiTaskClassifier
from fish_species.training.checkpoints import load_model_state_compat
from fish_species.training.checkpoints import build_checkpoint_payload, load_checkpoint, save_checkpoint
from fish_species.training.modes import get_profile
from fish_species.training.contrastive import (
    balanced_contrastive_loss,
    hierarchical_contrastive_loss,
)
from fish_species.training.losses import build_dual_species_criteria
from fish_species.training.stages import (
    apply_stage2_trainable_scope,
    initialise_species_classifier,
)


class _FeatureBackbone(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.features = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self._fish_is_feature_backbone = True
        self._fish_feature_dim = dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            inputs = layer(inputs)
        return inputs


def _model_config(**model_updates) -> dict:
    model = {
        "species_classifier": {"type": "cosine"},
        "prototype_classifier": {"enabled": False},
        "dual_species_classifier": {"enabled": False},
    }
    model.update(model_updates)
    return {"model": model, "multi_task": {}}


class CosineClassifierTests(unittest.TestCase):
    def test_shape_normalization_scale_gradient_and_checkpoint(self):
        classifier = CosineClassifier(2, 2, initial_scale=2, learnable_scale=True)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[10.0, 0.0], [0.0, 0.25]]))
        inputs = torch.tensor([[4.0, 0.0], [0.0, 3.0]], requires_grad=True)
        output = classifier(inputs)
        self.assertEqual(tuple(output.shape), (2, 2))
        torch.testing.assert_close(output, torch.eye(2) * 2.0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertIsNotNone(classifier.log_scale.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save(classifier.state_dict(), path)
            restored = CosineClassifier(2, 2)
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(inputs.detach()), output.detach())

    def test_fixed_scale_and_amp(self):
        classifier = CosineClassifier(3, 4, learnable_scale=False)
        self.assertNotIsInstance(classifier.log_scale, nn.Parameter)
        values = torch.randn(2, 3, requires_grad=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = classifier(values)
        result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(values.grad).all())


class PrototypeTests(unittest.TestCase):
    def test_calculation_zero_count_ema_and_fusion(self):
        module = PrototypeClassifier(2, 3, fusion_mode="fixed", learned_weight=0.25)
        module.rebuild([(torch.tensor([[2.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1]))])
        torch.testing.assert_close(module.prototypes[:2], torch.eye(2))
        torch.testing.assert_close(module.prototypes[2], torch.zeros(2))
        self.assertTrue(torch.isfinite(module(torch.randn(2, 2))).all())
        learned = torch.ones(1, 3)
        prototype = torch.full((1, 3), 3.0)
        torch.testing.assert_close(module.fuse(learned, prototype), torch.full((1, 3), 2.5))

        ema = PrototypeClassifier(2, 2, update="ema", momentum=0.5)
        ema.ema_update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
        ema.ema_update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))
        torch.testing.assert_close(
            ema.prototypes[0], F.normalize(torch.tensor([0.5, 0.5]), dim=0)
        )
        self.assertEqual(ema.counts.tolist(), [2, 0])

    def test_frequency_fusion_checkpoint_and_distributed_reduction(self):
        module = PrototypeClassifier(
            2, 2, fusion_mode="frequency_dependent", prototype_strength=10
        )
        module.counts.copy_(torch.tensor([0, 10]))
        learned = torch.tensor([[2.0, 2.0]])
        prototype = torch.tensor([[4.0, 4.0]])
        torch.testing.assert_close(module.fuse(learned, prototype), torch.tensor([[4.0, 3.0]]))
        restored = PrototypeClassifier(2, 2)
        restored.load_state_dict(module.state_dict())
        torch.testing.assert_close(restored.counts, module.counts)
        with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
            "torch.distributed.is_available", return_value=True
        ), mock.patch("torch.distributed.all_reduce") as reduction:
            module.accumulate(torch.eye(2), torch.tensor([0, 1]))
        self.assertEqual(reduction.call_count, 2)

    def test_checkpoint_payload_round_trip(self):
        module = PrototypeClassifier(2, 2)
        module.rebuild([(torch.eye(2), torch.tensor([0, 1]))])
        payload = build_checkpoint_payload(
            profile=get_profile("standard"), model_state=module.state_dict(),
            cfg={}, label_to_index_by_task={}, index_to_label_by_task={},
            best_val_score=1.0, selection_metric="species_accuracy", best_epoch=1,
            long_tail_metadata={"prototype_counts": module.counts.clone()},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(payload, path)
            restored = load_checkpoint(path, map_location="cpu")
        torch.testing.assert_close(
            restored["model_state"]["prototypes"], module.prototypes
        )
        torch.testing.assert_close(
            restored["long_tail_metadata"]["prototype_counts"], module.counts
        )


class StageAndDualHeadTests(unittest.TestCase):
    def test_prototype_initialization_and_exact_scopes(self):
        cfg = _model_config(prototype_classifier={"enabled": True})
        model = MultiTaskClassifier(_FeatureBackbone(), {"genus": 2, "species": 3}, cfg)
        model.prototype_classifier.set_from_sums(
            torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]),
            torch.ones(3, dtype=torch.long),
        )
        initialise_species_classifier(model, "prototype")
        torch.testing.assert_close(model.species_head.weight, model.prototype_classifier.prototypes)
        apply_stage2_trainable_scope(model, "heads")
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.species_head.parameters()))
        apply_stage2_trainable_scope(model, "heads_and_last_block")
        self.assertFalse(any(p.requires_grad for p in model.backbone.features[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.backbone.features[-1].parameters()))
        apply_stage2_trainable_scope(model, "full_model")
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_dual_outputs_losses_and_legacy_loading(self):
        cfg = _model_config(dual_species_classifier={
            "enabled": True, "classifier_type": "linear",
            "inference": {"mode": "fused", "natural_weight": 0.25},
        })
        model = MultiTaskClassifier(_FeatureBackbone(), {"genus": 2, "species": 3}, cfg)
        outputs = model(torch.randn(5, 4))
        for name in ("species_natural", "species_balanced", "species_dual_fused", "species"):
            self.assertEqual(tuple(outputs[name].shape), (5, 3))
        torch.testing.assert_close(
            outputs["species"], 0.25 * outputs["species_natural"] + 0.75 * outputs["species_balanced"]
        )
        import pandas as pd
        frame = pd.DataFrame({"species": ["a", "a", "b", "c"], "group": range(4)})
        natural, balanced = build_dual_species_criteria(
            frame, "species", {"a": 0, "b": 1, "c": 2}, "group",
            torch.device("cpu"), {"balanced_method": "logit_adjustment", "tau": 1},
        )
        labels = torch.tensor([0, 1, 2, 0, 1])
        self.assertTrue(torch.isfinite(natural(outputs["species_natural"], labels)))
        self.assertTrue(torch.isfinite(balanced(outputs["species_balanced"], labels)))

        legacy = {
            "heads.species.weight": torch.randn_like(model.natural_head.weight),
            "heads.species.bias": torch.randn_like(model.natural_head.bias),
        }
        missing, _ = load_model_state_compat(model, legacy)
        self.assertNotIn("natural_head.weight", missing)
        torch.testing.assert_close(model.natural_head.weight, model.balanced_head.weight)


class ContrastiveTests(unittest.TestCase):
    def test_hierarchical_pairs_and_no_valid_anchor(self):
        embeddings = torch.randn(4, 8, requires_grad=True)
        loss, stats = hierarchical_contrastive_loss(
            embeddings, torch.tensor([0, 0, 1, 2]), torch.tensor([0, 0, 0, 1])
        )
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertGreater(stats.positive_pairs, 2)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        no_loss, stats = hierarchical_contrastive_loss(
            torch.randn(2, 4), torch.tensor([0, 1]), torch.tensor([0, 1])
        )
        self.assertIsNone(no_loss)
        self.assertEqual(stats.valid_anchors, 0)

    def test_balanced_imbalance_singletons_and_prototypes(self):
        embeddings = torch.randn(7, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 2])
        prototypes = torch.randn(3, 6)
        loss, stats = balanced_contrastive_loss(
            embeddings, labels, prototype_embeddings=prototypes,
            prototype_counts=torch.ones(3, dtype=torch.long), class_average=True,
        )
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertEqual(stats.valid_anchors, 7)
        self.assertEqual(stats.prototype_positive_pairs, 7)
        self.assertTrue(torch.isfinite(embeddings.grad).all())
        no_loss, stats = balanced_contrastive_loss(
            torch.randn(3, 4), torch.arange(3), prototype_embeddings=None
        )
        self.assertIsNone(no_loss)
        self.assertEqual(stats.valid_anchors, 0)


class TaxonomyAndValidationTests(unittest.TestCase):
    def test_costs_metrics_and_efficient_risk_match_dense(self):
        mapping = torch.tensor([0, 0, 1, 1])
        target = torch.tensor([0, 0, 2])
        prediction = torch.tensor([0, 1, 0])
        self.assertEqual(taxonomic_costs(target, prediction, mapping).tolist(), [0, 1, 2])
        metrics = taxonomic_metrics(target, prediction, mapping)
        self.assertAlmostEqual(metrics["species_taxonomic_mean_cost"], 1.0)
        probs = torch.tensor([[0.35, 0.30, 0.34, 0.01]])
        efficient = minimum_taxonomic_risk_predictions(probs, mapping)
        dense = torch.empty(4, 4)
        for truth in range(4):
            dense[truth] = taxonomic_costs(
                torch.full((4,), truth), torch.arange(4), mapping
            )
        expected = (probs @ dense).argmin(1)
        torch.testing.assert_close(efficient, expected)

    def test_configuration_validation(self):
        root = Path(__file__).parents[1]
        cfg = load_config(root / "config.yaml")
        validate_config(cfg, check_paths=False, check_model_registry=False)
        cfg["model"]["species_classifier"]["type"] = "invalid"
        with self.assertRaises(ConfigValidationError):
            validate_config(cfg, check_paths=False, check_model_registry=False)

        base = load_config(root / "config.yaml")
        mutations = [
            lambda c: c["model"]["prototype_classifier"].update(momentum=1.0),
            lambda c: c["model"]["prototype_classifier"]["fusion"].update(learned_weight=1.5),
            lambda c: c["multi_task"]["hierarchical_contrastive"].update(enabled=True, temperature=0),
            lambda c: c["long_tail"]["staged_training"].update(trainable_scope="unknown"),
            lambda c: c["model"]["dual_species_classifier"]["inference"].update(mode="unknown"),
        ]
        for mutation in mutations:
            invalid = copy.deepcopy(base)
            mutation(invalid)
            with self.subTest(invalid=invalid):
                with self.assertRaises(ConfigValidationError):
                    validate_config(invalid, check_paths=False, check_model_registry=False)

        conflict = copy.deepcopy(base)
        conflict["model"]["dual_species_classifier"]["enabled"] = True
        conflict["training"]["sampling"]["strategy"] = "weighted"
        with self.assertRaises(ConfigValidationError):
            validate_config(conflict, check_paths=False, check_model_registry=False)


if __name__ == "__main__":
    unittest.main()
