from __future__ import annotations

import unittest

import pandas as pd

from src.splits import annotate_long_tail_groups, make_long_tail_splits


class LongTailSplitTests(unittest.TestCase):
    def setUp(self):
        counts = {"Head fish": 11, "Tail ten": 10, "Few fish": 3, "Pair fish": 2}
        self.frame = pd.DataFrame(
            {
                "species": [label for label, count in counts.items() for _ in range(count)],
                "image_id": [f"image-{index}" for index in range(sum(counts.values()))],
            }
        )

    def test_frequency_cohorts_use_full_labeled_counts(self):
        annotated = annotate_long_tail_groups(self.frame, "species", 11)
        groups = annotated.groupby("species")["__long_tail_group__"].first().to_dict()
        self.assertEqual(groups["Head fish"], "head")
        self.assertEqual(groups["Tail ten"], "tail")
        shots = annotated.groupby("species")["__shot_group__"].first().to_dict()
        self.assertEqual(shots["Few fish"], "few_2_to_5")
        self.assertEqual(shots["Tail ten"], "medium_6_to_20")

    def test_split_is_deterministic_and_retains_every_species_for_training(self):
        first = make_long_tail_splits(
            self.frame, "species", test_size=0.10, val_size=0.15, seed=7
        )
        second = make_long_tail_splits(
            self.frame, "species", test_size=0.10, val_size=0.15, seed=7
        )
        for left, right in zip(first, second):
            self.assertEqual(left["image_id"].tolist(), right["image_id"].tolist())
        self.assertEqual(set(first[0]["species"]), set(self.frame["species"]))
        self.assertEqual(len(first[0]) + len(first[1]) + len(first[2]), len(self.frame))


if __name__ == "__main__":
    unittest.main()

