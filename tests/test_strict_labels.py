from __future__ import annotations

import unittest

import pandas as pd

from fish_species.data.labels import build_label_maps


class StrictLabelTests(unittest.TestCase):
    def test_supervised_label_maps_reject_incomplete_rows(self):
        frame = pd.DataFrame({"genus": ["Salmo", "Salmo"], "species": ["Salmo salar", None]})
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_label_maps(frame, {"genus": "genus", "species": "species"})


if __name__ == "__main__":
    unittest.main()
