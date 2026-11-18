"""Partitions are deterministic, disjoint and complete."""

import unittest

import numpy as np
import pandas as pd

import _path  # noqa: F401

from fedheads.data.partition import check_partition, dirichlet_partition, domain_partition, iid_partition


def train_rows(n=360, seed=0, domains=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"uid": [f"s:train:{i}" for i in range(n)],
                       "label": rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])})
    if domains:
        df["domain"] = [domains[i % len(domains)] for i in range(n)]
    return df


class IID(unittest.TestCase):
    def test_deterministic(self):
        df = train_rows()
        self.assertEqual(iid_partition(df, "s", 5, 7), iid_partition(df, "s", 5, 7))

    def test_seed_changes_assignment(self):
        df = train_rows()
        self.assertNotEqual(iid_partition(df, "s", 5, 1), iid_partition(df, "s", 5, 2))

    def test_disjoint_complete_and_balanced(self):
        df = train_rows()
        part = iid_partition(df, "s", 5, 3)
        check_partition(part, df)
        sizes = [len(v) for v in part.values()]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(sorted(part), [f"s_c{k}" for k in range(5)])

    def test_label_mix_is_stratified(self):
        df = train_rows(1000)
        part = iid_partition(df, "s", 5, 0)
        lab = dict(zip(df["uid"], df["label"]))
        overall = df["label"].value_counts(normalize=True).sort_index().to_numpy()
        for uids in part.values():
            share = pd.Series([lab[u] for u in uids]).value_counts(normalize=True).sort_index().to_numpy()
            np.testing.assert_allclose(share, overall, atol=0.02)

    def test_check_partition_catches_overlap_and_gaps(self):
        df = train_rows(10)
        with self.assertRaises(AssertionError):
            check_partition({"a": ["s:train:0", "s:train:1"], "b": ["s:train:1"]}, df)
        with self.assertRaises(AssertionError):
            check_partition({"a": ["s:train:0"]}, df)


class Dirichlet(unittest.TestCase):
    def test_deterministic_complete_and_min_size(self):
        df = train_rows()
        a = dirichlet_partition(df, "s", 5, alpha=0.1, seed=4, min_size=20)
        self.assertEqual(a, dirichlet_partition(df, "s", 5, alpha=0.1, seed=4, min_size=20))
        check_partition(a, df)
        self.assertGreaterEqual(min(len(v) for v in a.values()), 20)

    def test_low_alpha_skews_labels_more_than_high_alpha(self):
        df = train_rows(1000)
        lab = dict(zip(df["uid"], df["label"]))

        def mean_max_share(alpha):
            part = dirichlet_partition(df, "s", 5, alpha=alpha, seed=0, min_size=20)
            return np.mean([pd.Series([lab[u] for u in v]).value_counts(normalize=True).max() for v in part.values()])

        self.assertGreater(mean_max_share(0.1), mean_max_share(100.0))

    def test_impossible_min_size_raises(self):
        with self.assertRaises(ValueError):
            dirichlet_partition(train_rows(50), "s", 5, alpha=1.0, seed=0, min_size=20)


class Domain(unittest.TestCase):
    DOMAINS = ["movie", "political", "subtitle", "tweet"]

    def test_each_client_holds_one_domain(self):
        df = train_rows(400, domains=self.DOMAINS)
        part = domain_partition(df, "mr", 5, seed=0)
        check_partition(part, df)
        dom = dict(zip(df["uid"], df["domain"]))
        self.assertEqual(len(part), 5)
        for cid, uids in part.items():
            self.assertEqual(len({dom[u] for u in uids}), 1, cid)
        self.assertEqual({dom[v[0]] for v in part.values()}, set(self.DOMAINS))
        self.assertIn("mr_movie_c0", part)

    def test_extra_client_goes_to_largest_domain(self):
        df = train_rows(400, domains=self.DOMAINS)
        df = pd.concat([df, train_rows(40, seed=1, domains=["political"]).assign(
            uid=[f"s:train:x{i}" for i in range(40)])], ignore_index=True)
        part = domain_partition(df, "mr", 5, seed=0)
        check_partition(part, df)
        self.assertEqual(sorted(part), ["mr_movie_c0", "mr_political_c0", "mr_political_c1",
                                        "mr_subtitle_c0", "mr_tweet_c0"])

    def test_too_few_clients_raises(self):
        with self.assertRaises(ValueError):
            domain_partition(train_rows(400, domains=self.DOMAINS), "mr", 3, seed=0)


if __name__ == "__main__":
    unittest.main()
