"""Modules, timing and the module runner."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _path  # noqa: F401

from test_engine import fake_table

from fedheads.runner.config import resolve
from fedheads.runner.guard import may_start
from fedheads.runner.timing import predict_hours
from fedheads.utils.io import CONFIG_DIR, REPO_ROOT, load_base_config, load_yaml

SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def module_files():
    return sorted((CONFIG_DIR / "modules").glob("*.yaml"))


class ModuleFiles(unittest.TestCase):
    def test_files_are_written_by_make_modules(self):
        r = subprocess.run([sys.executable, str(SCRIPTS / "make_modules.py"), "--check"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for p in module_files():
            self.assertTrue(p.read_text(encoding="utf-8").startswith("# Written by scripts/make_modules.py"), p.name)

    def test_runs_resolve_with_distinct_ids(self):
        from run_one import parse_sets

        for p in module_files():
            m = load_yaml(p)
            ids = [resolve(s["experiment"], s["method"], s["setting"], s["seed"], parse_sets(s["set"]))["run_id"]
                   for lane in m["lanes"] for s in lane]
            self.assertEqual(len(ids), len(set(ids)), p.name)

    def test_long_modules_are_refused(self):
        import make_modules

        runs = [make_modules.spec("E2", "fedavg", "iid12", s, ["fl.rounds=2000"]) for s in range(10)]
        with self.assertRaises(ValueError):
            make_modules.render("BIG", "too long", False, runs)

    def test_ada_modules_have_one_lane_and_fit_the_job(self):
        import make_modules

        m = load_yaml(CONFIG_DIR / "modules" / "ADA_TEST.yaml")
        self.assertEqual(m["platform"], "ada")
        self.assertEqual(len(m["lanes"]), 1)
        runs = [make_modules.spec("E2", "fedavg", "iid12", s, ["fl.rounds=300"]) for s in range(5)]
        with self.assertRaises(ValueError):
            make_modules.render("BIG", "too long for 6 h", False, runs, "ada")

    def test_tuning_grid_covers_every_head_once_with_numeric_rates(self):
        """6 heads x 4 head learning rates, FedAvg, seed 0, 100 rounds, on Ada."""
        from run_one import parse_sets
        from fedheads.models.heads import MAIN_HEADS

        seen = []
        for name in ("T1", "T2", "T3"):
            m = load_yaml(CONFIG_DIR / "modules" / f"{name}.yaml")
            self.assertEqual((m["platform"], len(m["lanes"])), ("ada", 1), name)
            for s in m["lanes"][0]:
                cfg = resolve(s["experiment"], s["method"], s["setting"], s["seed"], parse_sets(s["set"]))
                lr = cfg["train"]["lr_head"]
                self.assertIsInstance(lr, float, f"{name}: lr_head read as {type(lr).__name__}")
                self.assertEqual((cfg["method"], cfg["seed"], cfg["fl"]["rounds"]), ("fedavg", 0, 100))
                self.assertEqual(cfg["train"]["lr_lora"], 3.0e-4)
                seen.append((cfg["model"]["head"], lr))
        self.assertEqual(sorted(seen), sorted((h, lr) for h in MAIN_HEADS for lr in (3e-4, 1e-3, 3e-3, 1e-2)))


    def test_edge_runs_are_one_step_below_the_grid(self):
        """edge rule, after T1 and T3: CNN, BiLSTM and CNN-BiLSTM at 1e-4."""
        from run_one import parse_sets

        m = load_yaml(CONFIG_DIR / "modules" / "T4.yaml")
        got = []
        for s in m["lanes"][0]:
            cfg = resolve(s["experiment"], s["method"], s["setting"], s["seed"], parse_sets(s["set"]))
            self.assertEqual((cfg["method"], cfg["fl"]["rounds"]), ("fedavg", 100))
            got.append((cfg["model"]["head"], cfg["train"]["lr_head"]))
        self.assertEqual(sorted(got), [("bilstm", 1e-4), ("cnn", 1e-4), ("cnn_bilstm", 1e-4)])

    def test_experiment_2_has_every_head_method_and_seed_once_at_200_rounds(self):
        """6 heads x 3 methods x 3 seeds, tuned head rate, 200 rounds; each module fits the guard."""
        from run_one import parse_sets
        import make_modules

        tuned_path = CONFIG_DIR / "tuned.yaml"
        if not tuned_path.exists():
            self.skipTest("configs/tuned.yaml not written yet")
        tuned = load_yaml(tuned_path)
        session = load_base_config()["session_ada"]
        seen = []
        for p in sorted((CONFIG_DIR / "modules").glob("E2_*.yaml")):
            m = load_yaml(p)
            self.assertEqual((m["platform"], len(m["lanes"])), ("ada", 1), p.name)
            elapsed = make_modules.ADA_SETUP_HOURS
            for s in m["lanes"][0]:
                cfg = resolve(s["experiment"], s["method"], s["setting"], s["seed"], parse_sets(s["set"]))
                self.assertEqual(cfg["fl"]["rounds"], 200)
                head = cfg["model"]["head"]
                want = float(tuned[head]["all"][0].split("=")[1])
                self.assertEqual(cfg["train"]["lr_head"], want, head)
                pred = predict_hours(cfg, make_modules.CALIB)
                self.assertTrue(may_start(elapsed, pred, session), f"{p.name}: {cfg['run_id']} would be held by the guard")
                elapsed += pred
                seen.append((head, cfg["method"], cfg["seed"]))
        self.assertEqual(sorted(seen), sorted((h, m, s) for h in make_modules.MAIN_HEADS_ORDER
                                              for m in ("fedavg", "local_only", "centralized") for s in (0, 1, 2)))

class AdaSpeeds(unittest.TestCase):
    def test_runner_without_benchmark_uses_the_speeds_make_modules_planned_with(self):
        """an Ada module with calibrate false must not fall back to planning speeds."""
        import make_modules
        import run_module

        self.assertIsNotNone(make_modules.CALIB)
        self.assertEqual(run_module.ada_calibration(), make_modules.CALIB)
        cfg = resolve("E1", "fedavg", "iid12", 0, {"model": {"head": "bilstm"}})
        self.assertLess(predict_hours(cfg, run_module.ada_calibration()), predict_hours(cfg, None))


class AdaGuard(unittest.TestCase):
    def test_no_run_can_end_after_five_and_a_half_hours(self):
        session = load_base_config()["session_ada"]
        self.assertEqual(session["guard_line_hours"], 5.5)
        for elapsed in [x / 10 for x in range(0, 70)]:
            for predicted in (0.1, 0.3, 0.5, 1.0, 2.0):
                if may_start(elapsed, predicted, session):
                    self.assertLessEqual(elapsed + session["run_time_safety_factor"] * predicted, 5.5)


class Timing(unittest.TestCase):
    def test_slower_head_and_more_rounds_take_longer(self):
        cfg = resolve("E2", "fedavg", "iid12", 0)
        calib = {"train_seq_per_s": 240.0, "eval_seq_per_s": 1800.0, "train_seq_per_s_by_head": {"linear": 240.0, "heavy": 120.0}}
        heavy = resolve("E2", "fedavg", "iid12", 0, {"model": {"head": "heavy"}})
        self.assertGreater(predict_hours(heavy, calib), predict_hours(cfg, calib))
        longer = resolve("E2", "fedavg", "iid12", 0, {"fl": {"rounds": 200}})
        self.assertGreater(predict_hours(longer), predict_hours(cfg))

    def test_planning_estimate_by_hand(self):
        # 100 rounds x 12 clients x 25 steps x 16 = 480,000 sequences at 150/s = 3,200 s;
        # 20 validation passes of 10,400 and 2 test passes of 20,703 at 500/s = 498.8 s; 120 s loading.
        cfg = resolve("E2", "fedavg", "iid12", 0)
        self.assertAlmostEqual(predict_hours(cfg), (3200 + 498.8 + 120) / 3600, places=3)

    def test_local_only_evaluates_per_client(self):
        fed = predict_hours(resolve("E2", "fedavg", "iid12", 0))
        loc = predict_hours(resolve("E2", "local_only", "iid12", 0))
        self.assertGreater(loc, fed)


class GuardLine(unittest.TestCase):
    def test_no_run_can_end_after_ten_and_a_half_hours(self):
        session = load_base_config()["session"]
        self.assertEqual(session["guard_line_hours"], 10.5)
        for elapsed in [x / 10 for x in range(0, 120)]:
            for predicted in (0.1, 0.3, 0.5, 1.0, 2.0):
                if may_start(elapsed, predicted, session):
                    self.assertLessEqual(elapsed + session["run_time_safety_factor"] * predicted, 10.5)


def write_tiny_backbone(path: Path):
    from transformers import XLMRobertaConfig, XLMRobertaModel

    cfg = XLMRobertaConfig(vocab_size=60, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=64, pad_token_id=1)
    XLMRobertaModel(cfg, add_pooling_layer=False).save_pretrained(path)


def write_tiny_tokens(tok_dir: Path):
    (tok_dir / "xlmr").mkdir(parents=True)
    for i, lang in enumerate(("mr", "en", "bn", "hi")):
        fake_table(lang, 90, i).to_parquet(tok_dir / "xlmr" / f"{lang}.parquet")
    (tok_dir / "tokens_report.json").write_text(json.dumps({"backbones": {"xlmr": {"pad_id": 1}}}), encoding="utf-8")


class RunModuleEndToEnd(unittest.TestCase):
    """The TEST module through run_module.py: benchmark, then FedAvg, centralized and local-only,
    each in its own process, on a tiny saved backbone and fake token tables, on the CPU."""

    def test_test_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_tiny_backbone(tmp / "backbone")
            write_tiny_tokens(tmp / "tokens")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            cmd = [sys.executable, str(SCRIPTS / "run_module.py"), "--module", "TEST", "--tok-dir", str(tmp / "tokens"),
                   "--out-dir", str(tmp / "out"), "--extra-set", f"model.backbones.xlmr={(tmp / 'backbone').as_posix()}"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
            report = json.loads((tmp / "out" / "module_report.json").read_text(encoding="utf-8"))
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            self.assertEqual(report["summary"], {"done": 3}, report)
            self.assertIn("linear", report["calibration"]["calibration"]["train_seq_per_s_by_head"])
            for row in report["runs"]:
                run = tmp / "out" / "runs" / row["run_id"]
                for f in ("config.resolved.yaml", "env.json", "summary.json", "final_test.csv"):
                    self.assertTrue((run / f).exists(), f"{row['run_id']}: {f}")
                s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(set(s["test_best_by_language"]), {"mr", "en", "bn", "hi"})
            # Resubmitting the same module skips the finished runs without starting them.
            r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
            self.assertEqual(r2.returncode, 0, r2.stdout[-2000:] + r2.stderr[-2000:])
            again = json.loads((tmp / "out" / "module_report.json").read_text(encoding="utf-8"))
            self.assertEqual([row.get("skipped") for row in again["runs"]], ["done earlier"] * 3)

    def test_missing_token_tables_stop_the_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "tokens").mkdir()
            r = subprocess.run([sys.executable, str(SCRIPTS / "run_module.py"), "--module", "TEST", "--tok-dir",
                                str(tmp / "tokens"), "--out-dir", str(tmp / "out")], capture_output=True, text=True)
            self.assertEqual(r.returncode, 1)
            report = json.loads((tmp / "out" / "module_report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["missing_token_tables"]), 4)


if __name__ == "__main__":
    unittest.main()
