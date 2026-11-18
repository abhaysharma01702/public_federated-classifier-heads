"""the two-class appendix (neutral removed)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedheads.data.loaders import drop_neutral  # noqa: E402
from fedheads.eval.metrics import macro_f1  # noqa: E402
from fedheads.models.heads import build_head  # noqa: E402
from fedheads.runner.config import resolve  # noqa: E402


class TwoClassTest(unittest.TestCase):
    def test_drop_neutral_maps_labels(self):
        t = pd.DataFrame({"uid": list("abcdef"), "split": ["train"] * 3 + ["test"] * 3,
                          "label": [0, 1, 2, 2, 1, 0]})
        out = drop_neutral({"bn": t})["bn"]
        self.assertEqual(out["uid"].tolist(), ["a", "c", "d", "f"])
        self.assertEqual(out["label"].tolist(), [0, 1, 1, 0])
        self.assertEqual(out.index.tolist(), [0, 1, 2, 3])
        self.assertEqual(t["label"].tolist(), [0, 1, 2, 2, 1, 0])  # input untouched

    def test_macro_f1_two_and_three_classes(self):
        y, p = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0])
        self.assertAlmostEqual(macro_f1(y, p, 2), (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(macro_f1(y, p), (0.8 + 2 / 3 + 0.0) / 3)  # default unchanged

    def test_heads_build_with_two_classes(self):
        import torch
        for name in ["linear", "cnn", "mlp", "attention", "bilstm", "cnn_bilstm"]:
            h = build_head(name, 32, 2)
            logits, _ = h(torch.randn(3, 7, 32), torch.ones(3, 7, dtype=torch.long))
            self.assertEqual(tuple(logits.shape), (3, 2), name)

    def test_setting_resolves_and_three_class_ids_unchanged(self):
        c = resolve("E8", "fedavg", "iid12_2c", 0, {})
        self.assertTrue(c["data"]["drop_neutral"])
        self.assertEqual(c["model"]["n_classes"], 2)
        base = resolve("E2", "fedavg", "iid12", 0, {})
        self.assertNotIn("n_classes", base["model"])
        self.assertFalse(base["data"].get("drop_neutral", False))


if __name__ == "__main__":
    unittest.main()
