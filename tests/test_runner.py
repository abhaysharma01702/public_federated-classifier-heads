"""Runner: session guard, config identity, client setup."""

import unittest

import _path  # noqa: F401

from test_engine import fake_table

from fedheads.runner.config import resolve
from fedheads.runner.guard import may_start
from fedheads.runner.setup import build_clients, eval_rows

SESSION = {"guard_line_hours": 11.25, "run_time_safety_factor": 1.2, "end_margin_hours": 0.5}   # formula check; real values in test_modules


class Guard(unittest.TestCase):
    def test_never_crosses_the_guard_line(self):
        for elapsed in [x / 4 for x in range(0, 48)]:
            for predicted in (0.1, 0.5, 1.0, 2.0, 5.0):
                if may_start(elapsed, predicted, SESSION):
                    self.assertLessEqual(elapsed + 1.2 * predicted + 0.5, 11.25)

    def test_examples(self):
        self.assertTrue(may_start(0.3, 1.0, SESSION))
        self.assertFalse(may_start(10.0, 1.0, SESSION))   # 10 + 1.2 + 0.5 > 11.25
        self.assertTrue(may_start(9.5, 1.0, SESSION))     # 9.5 + 1.2 + 0.5 = 11.2


class Identity(unittest.TestCase):
    def test_run_id_is_stable_and_changes_with_config(self):
        a = resolve("E2", "fedavg", "iid12", 0)
        b = resolve("E2", "fedavg", "iid12", 0)
        c = resolve("E2", "fedavg", "iid12", 0, {"train": {"lr_lora": 1e-4}})
        self.assertEqual(a["run_id"], b["run_id"])
        self.assertNotEqual(a["run_id"], c["run_id"])
        self.assertTrue(a["run_id"].startswith("E2-iid12-fedavg-linear-s0-"))

    def test_method_layers(self):
        self.assertEqual(resolve("E2", "fedavg", "iid12", 0)["strategy"]["mode"], "federated")
        self.assertEqual(resolve("E2", "centralized", "iid12", 0)["strategy"]["mode"], "centralized")
        self.assertEqual(resolve("E2", "local_only", "iid12", 0)["strategy"]["mode"], "local_only")

    def test_main_setting_has_twelve_clients(self):
        cfg = resolve("E2", "fedavg", "iid12", 0)
        self.assertEqual(sum(cfg["setting_cfg"]["clients"].values()), 12)
        self.assertEqual(set(cfg["setting_cfg"]["clients"]), set(cfg["data"]["languages"]))


class Setup(unittest.TestCase):
    def tables(self):
        return {"mr": fake_table("mr", 300, 1, domains=["movie", "political", "subtitle", "tweet"]),
                "en": fake_table("en", 150, 2)}

    def cfg(self, setting):
        return {"seed": 0, "data": {"languages": ["mr", "en"]}, "setting_cfg": setting}

    def test_one_language_per_client_and_all_training_rows_used(self):
        tables = self.tables()
        for setting in ({"partition": "iid", "clients": {"mr": 3, "en": 2}},
                        {"partition": "dirichlet", "alpha": 0.5, "min_size": 5, "clients": {"mr": 3, "en": 2}},
                        {"partition": "domain", "domain_languages": ["mr"], "clients": {"mr": 4, "en": 2}}):
            with self.subTest(partition=setting["partition"]):
                clients = build_clients(tables, self.cfg(setting))
                self.assertEqual(len(clients), sum(setting["clients"].values()))
                for lang, t in tables.items():
                    mine = [c for c in clients if c.language == lang]
                    rows = sorted(r for c in mine for r in c.rows)
                    self.assertEqual(rows, sorted(t.index[t["split"] == "train"].tolist()))

    def test_domain_clients_hold_one_domain(self):
        tables = self.tables()
        clients = build_clients(tables, self.cfg({"partition": "domain", "domain_languages": ["mr"],
                                                  "clients": {"mr": 4, "en": 1}}))
        for c in clients:
            if c.language == "mr":
                self.assertEqual(c.table["domain"].iloc[c.rows].nunique(), 1)

    def test_eval_rows_are_the_official_splits(self):
        tables = self.tables()
        rows = eval_rows(tables, self.cfg({}))
        for lang, t in tables.items():
            for split in ("validation", "test"):
                self.assertTrue((t["split"].iloc[rows[split][lang]] == split).all())
                self.assertEqual(len(rows[split][lang]), int((t["split"] == split).sum()))


if __name__ == "__main__":
    unittest.main()
