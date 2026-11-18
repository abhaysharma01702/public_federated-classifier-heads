"""FedProx and FedAdam: the math, exact resume, and that they change training."""

import tempfile
import unittest

import torch

import _path  # noqa: F401

from test_engine import TINY_CFG, tiny_clients, tiny_model, tiny_split_rows, tiny_tables

from fedheads.eval.metrics import Evaluator
from fedheads.fl.algorithms import add_prox_grads, server_adam_step
from fedheads.fl.server import FederatedRun
from fedheads.runner.config import resolve

PROX = {**TINY_CFG, "strategy": {"name": "fedprox", "mode": "federated", "prox_mu": 0.5}}
ADAM = {**TINY_CFG, "strategy": {"name": "fedadam", "mode": "federated",
                                 "server": {"name": "adam", "lr": 1e-2, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3}}}


def setup(cfg, tmp):
    tables = tiny_tables()
    ev = Evaluator(tables, tiny_split_rows(tables), pad_id=1, device=torch.device("cpu"), batch_size=8)
    return FederatedRun(cfg, tiny_model(), tiny_clients(tables), ev, tmp, torch.device("cpu"), 1,
                        lambda m: None, use_amp=False)


class Math(unittest.TestCase):
    def test_fedadam_first_step(self):
        x = {"w": torch.tensor([0.0, 1.0])}
        mean = {"w": torch.tensor([0.5, 0.0])}
        new, opt = server_adam_step(x, mean, None, {"lr": 0.01, "tau": 1e-3})
        # m = 0.1 delta, v = 0.01 delta^2: the step is lr * delta / (|delta| + 10 tau)
        torch.testing.assert_close(new["w"], torch.tensor([0.01 * 0.5 / 0.51, 1.0 - 0.01 * 1.0 / 1.01]))
        torch.testing.assert_close(opt["m"]["w"], torch.tensor([0.05, -0.1]))

    def test_prox_gradient_points_back_to_the_anchor(self):
        p = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        p.grad = torch.zeros(2)
        add_prox_grads([("w", p)], {"w": torch.tensor([0.0, 0.0])}, mu=0.1)
        torch.testing.assert_close(p.grad, torch.tensor([0.2, -0.1]))


class Runs(unittest.TestCase):
    def test_exact_resume_and_different_from_fedavg(self):
        torch.use_deterministic_algorithms(True)
        try:
            with tempfile.TemporaryDirectory() as f:
                fedavg = setup(TINY_CFG, f).run()["state"]
            for cfg in (PROX, ADAM):
                with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                    straight = setup(cfg, a).run()
                    setup(cfg, b).run(stop_after_round=2)
                    resumed = setup(cfg, b).run()
                    for k in straight["state"]:
                        torch.testing.assert_close(straight["state"][k], resumed["state"][k], atol=0, rtol=0)
                    diff = sum(float((straight["state"][k] - fedavg[k]).abs().sum()) for k in fedavg)
                    self.assertGreater(diff, 0.0, cfg["strategy"]["name"])
        finally:
            torch.use_deterministic_algorithms(False)

    def test_method_configs_resolve(self):
        p = resolve("E3", "fedprox", "iid12", 0)
        self.assertEqual(p["strategy"]["prox_mu"], 0.01)
        a = resolve("E3", "fedadam", "iid12", 0)
        self.assertEqual(a["strategy"]["server"]["name"], "adam")
        self.assertNotEqual(p["run_id"], a["run_id"])


if __name__ == "__main__":
    unittest.main()
