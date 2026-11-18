"""Data preparation: labels by name, conflicts, duplicates and overlap (scripts/prepare_data.py)."""

import sys
import unittest

import pandas as pd

import _path  # noqa: F401
from fedheads.utils.io import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_data as pdx  # noqa: E402
from tokenize_data import head_tail  # noqa: E402


class Labels(unittest.TestCase):
    def test_mapped_by_name(self):
        f = pdx.frame(["a", "b", "c"], [-1, 0, 1], pdx.NAMES["mahasent"], "train", "movie")
        self.assertEqual(f["label"].tolist(), [0, 1, 2])
        f = pdx.frame(["a", "b", "c"], [0, 1, 2], pdx.NAMES["sentnob"], "train", "comment")
        self.assertEqual(f["label"].tolist(), [1, 2, 0])   # SentNoB: 0 neutral, 1 positive, 2 negative

    def test_unknown_label_stops(self):
        with self.assertRaises(ValueError):
            pdx.frame(["a"], [7], pdx.NAMES["iitp"], "train", "movie")


class Cleaning(unittest.TestCase):
    def test_rules(self):
        rows = [("train", 0, "same  text"), ("train", 0, "same text"),        # duplicate after whitespace
                ("train", 1, "clash"), ("train", 2, "clash"),                  # conflicting labels
                ("train", 1, "seen in test"), ("test", 1, "seen in test"),     # overlap
                ("train", 2, "keep"), ("validation", 0, "v"), ("train", 0, "  ")]
        df = pd.DataFrame(rows, columns=["split", "label", "text"]).assign(domain="d")
        out, rep = pdx.clean(df)
        train = sorted(out[out["split"] == "train"]["text"])
        self.assertEqual(train, ["keep", "same text"])
        self.assertEqual(rep["train_duplicate"], 1)
        self.assertEqual(rep["train_in_validation_or_test"], 1)
        self.assertEqual(rep["conflicting_label"], {"train": 2})
        self.assertEqual(out["uid"].is_unique, True)
        self.assertEqual(sorted(out[out["split"] != "train"]["text"]), ["seen in test", "v"])


class Truncation(unittest.TestCase):
    def test_head_and_tail_kept(self):
        ids = [0] + list(range(100, 400)) + [2]          # <s>, 300 tokens, </s>
        out = head_tail(ids, 128, 64)
        self.assertEqual(len(out), 128)
        self.assertEqual(out[:64], ids[:64])
        self.assertEqual(out[64:], ids[-64:])
        self.assertEqual((out[0], out[-1]), (0, 2))

    def test_short_unchanged(self):
        self.assertEqual(head_tail([0, 5, 2], 128, 64), [0, 5, 2])


if __name__ == "__main__":
    unittest.main()
