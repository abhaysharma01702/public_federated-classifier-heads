"""Centralized and local-only baselines on a tiny model: exact resume and the defining property of each."""

import tempfile
import unittest

import torch

import _path  # noqa: F401

from test_engine import TINY_CFG, tiny_clients, tiny_model, tiny_split_rows, tiny_tables

from fedheads.eval.metrics import Evaluator
from fedheads.fl.baselines import CentralizedRun, LocalOnlyRun
from fedheads.fl.server import FederatedRun


def make(cls, tmp):
    tables = tiny_tables()
    ev = Evaluator(tables, tiny_split_rows(tables), pad_id=1, device=torch.device("cpu"), batch_size=8)
    return cls(TINY_CFG, tiny_model(), tiny_clients(tables), ev, tmp, torch.device("cpu"), 1,
               lambda m: None, use_amp=False)


class Centralized(unittest.TestCase):
    def test_resume_equivalence(self):
        torch.use_deterministic_algorithms(True)
        try:
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                straight = make(CentralizedRun, a).run()
                make(CentralizedRun, b).run(stop_after_round=2)
                resumed = make(CentralizedRun, b).run()
                for k in straight["state"]:
                    torch.testing.assert_close(straight["state"][k], resumed["state"][k], atol=0, rtol=0)
        finally:
            torch.use_deterministic_algorithms(False)

    def test_same_sequences_as_fedavg_and_nothing_sent(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            cen = make(CentralizedRun, a).run()["logs"]
            fed = make(FederatedRun, b).run()["logs"]
            self.assertEqual(sum(r["sequences"] for r in cen["timing"]), sum(r["sequences"] for r in fed["timing"]))
            self.assertEqual(sum(r["bytes_up"] for r in cen["comm"]), 0)
            self.assertGreater(sum(r["bytes_up"] for r in fed["comm"]), 0)


class LocalOnly(unittest.TestCase):
    def test_every_client_trains_and_results_average_per_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = make(LocalOnlyRun, tmp)
            res = run.run()
            self.assertEqual(len(res["done"]), 4)
            rows = LocalOnlyRun.final_test(res)
            # every client's best and final weights are saved.
            from pathlib import Path
            saved = sorted(p.name for p in (Path(tmp) / "client_states").glob("*.pt"))
            self.assertEqual(len(saved), 2 * len(run.clients))
            self.assertIn(f"{run.clients[0].client_id}_best_state.pt", saved)
            self.assertEqual({(r["state"], r["language"]) for r in rows},
                             {(s, lang) for s in ("best", "final") for lang in ("mr", "en")})
            self.assertTrue(all(r["clients"] == 2 for r in rows))
            self.assertTrue(0.0 <= LocalOnlyRun.validation_score(res) <= 1.0)

    def test_resume_skips_finished_clients(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = make(LocalOnlyRun, tmp).run()
            again = make(LocalOnlyRun, tmp).run()   # every client already in the checkpoint
            self.assertEqual(first["done"].keys(), again["done"].keys())
            self.assertEqual(len(again["timing"]), len(first["timing"]))


if __name__ == "__main__":
    unittest.main()
