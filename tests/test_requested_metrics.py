from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fish_species.training.epochs import run_hierarchy_epoch


class _MetricDataset(Dataset):
    species = [0, 1, 2, 3]
    genus = [0, 0, 1, 1]

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "image": torch.tensor([index], dtype=torch.float32),
            "labels": {
                "genus": torch.tensor(self.genus[index]),
                "species": torch.tensor(self.species[index]),
            },
        }


class _MetricModel(nn.Module):
    def forward(self, images):
        indices = images[:, 0].long()
        species = torch.full((len(indices), 6), -3.0)
        genus = torch.full((len(indices), 2), -3.0)
        for row, index in enumerate(indices.tolist()):
            predicted_species = index if index != 3 else 2
            species[row, predicted_species] = 3.0
            genus[row, 0 if predicted_species < 2 else 1] = 3.0
        return {"genus": genus, "species": species}


class RequestedMetricTests(unittest.TestCase):
    def test_species_and_taxonomy_metric_surface(self):
        loader = DataLoader(_MetricDataset(), batch_size=2, shuffle=False)
        metrics, _, _ = run_hierarchy_epoch(
            _MetricModel(),
            loader,
            {"genus": nn.CrossEntropyLoss(), "species": nn.CrossEntropyLoss()},
            None,
            torch.device("cpu"),
            False,
            use_amp=False,
            metric_context={
                "index_to_label_by_task": {
                    "genus": {0: "Alpha", 1: "Beta"},
                    "species": {
                        0: "Alpha one", 1: "Alpha two", 2: "Beta one",
                        3: "Beta two", 4: "Gamma one", 5: "Gamma two",
                    },
                },
                "species_counts": {
                    "Alpha one": 3,
                    "Alpha two": 8,
                    "Beta one": 12,
                    "Beta two": 30,
                },
            },
        )
        for key in (
            "species_macro_f1",
            "species_balanced_accuracy",
            "species_top1_accuracy",
            "species_top5_accuracy",
            "genus_macro_f1",
            "genus_species_consistency",
            "species_few_shot_2_to_5_macro_f1",
            "species_medium_shot_6_to_20_macro_f1",
            "species_many_shot_over_20_macro_f1",
            "species_head_over_10_macro_f1",
            "species_tail_10_or_fewer_macro_f1",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["species_top1_accuracy"], 0.75)
        self.assertEqual(metrics["species_top5_accuracy"], 1.0)
        self.assertEqual(metrics["genus_species_consistency"], 1.0)


if __name__ == "__main__":
    unittest.main()
