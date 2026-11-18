"""Training engine, on a tiny random XLM-R on the CPU."""

import os
import tempfile
import unittest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import pandas as pd
import torch

import _path  # noqa: F401

from fedheads.data.loaders import ClientData, pad
from fedheads.eval.metrics import Evaluator
from fedheads.fl.server import FederatedRun, sampling_schedule, state_bytes, weighted_average
from fedheads.models.heads import build_head, masked_mean
from fedheads.models.model import build_model, count_parameters, get_trainable_state, set_trainable_state

LORA = {"r": 4, "alpha": 8, "dropout": 0.1, "target_modules": ["query", "value"]}
MODEL_CFG = {"lora": LORA, "dropout": 0.1, "head": "linear"}


def tiny_encoder(seed=0):
    from transformers import XLMRobertaConfig, XLMRobertaModel

    torch.manual_seed(seed)
    cfg = XLMRobertaConfig(vocab_size=60, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=64, pad_token_id=1)
    return XLMRobertaModel(cfg, add_pooling_layer=False)


def tiny_model(seed=0):
    return build_model(MODEL_CFG, seed=seed, encoder=tiny_encoder())


def fake_table(language, n, seed, domains=None):
    """Rows cycle through train, validation and test; optional domains cycle too."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        length = int(rng.integers(3, 12))
        ids = np.concatenate([[0], rng.integers(5, 60, size=length - 2), [2]]).astype(np.int32)
        row = {"uid": f"{language}:{i}", "split": ["train", "validation", "test"][i % 3],
               "label": int(rng.integers(0, 3)), "ids": ids}
        if domains:
            row["domain"] = domains[(i // 3) % len(domains)]
        rows.append(row)
    return pd.DataFrame(rows)


def tiny_tables():
    return {"mr": fake_table("mr", 60, 1, domains=["movie", "tweet"]), "en": fake_table("en", 60, 2)}


def tiny_clients(tables):
    clients = []
    for lang, t in tables.items():
        train = np.flatnonzero(t["split"].to_numpy() == "train")
        for k, part in enumerate(np.array_split(train, 2)):
            clients.append(ClientData(f"{lang}_c{k}", lang, t, part))
    return clients


def tiny_split_rows(tables):
    return {s: {lang: np.flatnonzero(t["split"].to_numpy() == s) for lang, t in tables.items()}
            for s in ("validation", "test")}


TINY_CFG = {"seed": 0,
            "train": {"batch_sentences": 4, "local_steps": 3, "lr_lora": 1e-3, "lr_head": 1e-2,
                      "weight_decay": 0.01, "grad_clip": 1.0},
            "fl": {"rounds": 4, "clients_per_round": 3, "eval_every": 2, "checkpoint_every": 2}}


def tiny_setup(tmp):
    tables = tiny_tables()
    ev = Evaluator(tables, tiny_split_rows(tables), pad_id=1, device=torch.device("cpu"), batch_size=8)
    return FederatedRun(TINY_CFG, tiny_model(), tiny_clients(tables), ev, tmp, torch.device("cpu"), 1,
                        lambda m: None, use_amp=False)


class MaskedPooling(unittest.TestCase):
    """Padding never changes the pooled output or the logits."""

    def test_padding_invariance(self):
        model = tiny_model().eval()
        seq = np.array([0, 7, 8, 9, 2], dtype=np.int32)
        ids1, m1 = pad([seq], 1)
        ids2, m2 = pad([seq, np.arange(0, 12, dtype=np.int32)], 1)  # forces 7 padding positions
        with torch.no_grad():
            l1, h1 = model(ids1, m1)
            l2, h2 = model(ids2, m2)
        torch.testing.assert_close(l1[0], l2[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(h1[0], h2[0], atol=1e-5, rtol=1e-5)

    def test_masked_mean_ignores_padding_values(self):
        h = torch.tensor([[[1.0], [3.0], [100.0]]])
        self.assertAlmostEqual(float(masked_mean(h, torch.tensor([[1, 1, 0]]))), 2.0)

    def test_unknown_head_is_refused(self):
        with self.assertRaises(ValueError):
            build_head("cnn_bilstm_typo", 16)


class Aggregation(unittest.TestCase):
    """Weighted FedAvg equals the hand-computed mean; only trainable keys exist."""

    def test_weighted_mean(self):
        a = {"w": torch.tensor([1.0, 2.0])}
        b = {"w": torch.tensor([3.0, 6.0])}
        out = weighted_average([a, b], [1, 3])
        torch.testing.assert_close(out["w"], torch.tensor([2.5, 5.0]))

    def test_frozen_weights_never_in_state(self):
        state = get_trainable_state(tiny_model())
        self.assertTrue(state)
        self.assertTrue(all(("lora_" in k) or k.startswith("head.") for k in state))
        self.assertFalse(any("embeddings" in k for k in state))

    def test_schedule_is_deterministic_without_repeats(self):
        s1, s2 = sampling_schedule(24, 12, 100, 0), sampling_schedule(24, 12, 100, 0)
        self.assertTrue(np.array_equal(s1, s2))
        self.assertTrue(all(len(set(r)) == 12 for r in s1))
        self.assertFalse(np.array_equal(s1, sampling_schedule(24, 12, 100, 1)))

    def test_bytes_are_fp32_count_of_trainable(self):
        model = tiny_model()
        state = get_trainable_state(model)
        self.assertEqual(state_bytes(state), 4 * count_parameters(model)["trainable"])


class Accessors(unittest.TestCase):
    """Accessors return exactly LoRA and head."""

    def test_roundtrip_and_only_trainable(self):
        model = tiny_model()
        state = get_trainable_state(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(sum(t.numel() for t in state.values()), n_trainable)
        changed = {k: v + 1.0 for k, v in state.items()}
        set_trainable_state(model, changed)
        for k, v in get_trainable_state(model).items():
            torch.testing.assert_close(v, changed[k])
        with self.assertRaises(KeyError):
            set_trainable_state(model, {"head.linear.weight": state["head.linear.weight"]})

    def test_tiny_parameter_counts(self):
        c = count_parameters(tiny_model())
        self.assertEqual(c["lora"], 2 * 2 * (16 * 4 + 4 * 16))   # 2 layers x (query, value) x (A + B)
        self.assertEqual(c["head"], 16 * 3 + 3)

    def test_xlmr_base_parameter_counts(self):
        try:
            from transformers import AutoConfig, AutoModel

            cfg = AutoConfig.from_pretrained("xlm-roberta-base", local_files_only=True)
        except Exception:
            self.skipTest("xlm-roberta-base config not cached")
        torch.manual_seed(0)
        enc = AutoModel.from_config(cfg, add_pooling_layer=False)
        model = build_model({"lora": {"r": 8, "alpha": 16, "dropout": 0.1, "target_modules": ["query", "value"]},
                             "dropout": 0.1, "head": "linear"}, seed=0, encoder=enc)
        c = count_parameters(model)
        self.assertEqual(c["lora"], 12 * 2 * (768 * 8 + 8 * 768))   # 294,912
        self.assertEqual(c["head"], 768 * 3 + 3)                    # 2,307
        self.assertGreater(c["frozen"], 270_000_000)


class TorchaoGuard(unittest.TestCase):
    """Kaggle's PEFT 0.20 + torchao 0.10 raised ImportError while adding LoRA."""

    def test_old_torchao_is_ignored_and_new_is_left_alone(self):
        import peft.import_utils as iu

        from fedheads.models.model import disable_incompatible_torchao

        if not hasattr(iu, "is_torchao_available"):
            self.skipTest("this PEFT version has no torchao check")
        original = iu.is_torchao_available
        try:
            self.assertIsNone(disable_incompatible_torchao("0.16.1"))
            self.assertIs(iu.is_torchao_available, original)
            note = disable_incompatible_torchao("0.10.0")
            self.assertIn("0.10.0", note)
            self.assertFalse(iu.is_torchao_available())
            tiny_model()  # LoRA still attaches
        finally:
            iu.is_torchao_available = original


class CrashResume(unittest.TestCase):
    """Straight rounds equal a killed-and-resumed run, exactly, on the CPU."""

    def test_resume_equivalence(self):
        torch.use_deterministic_algorithms(True)
        try:
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                straight = tiny_setup(a).run()
                tiny_setup(b).run(stop_after_round=2)      # "killed" after the round-2 checkpoint
                resumed = tiny_setup(b).run()             # fresh process state, resumes at round 2
                for k in straight["state"]:
                    torch.testing.assert_close(straight["state"][k], resumed["state"][k], atol=0, rtol=0)
                self.assertEqual(straight["best"]["round"], resumed["best"]["round"])
                self.assertEqual(len(straight["logs"]["metrics"]), len(resumed["logs"]["metrics"]))
        finally:
            torch.use_deterministic_algorithms(False)

    def test_training_changes_the_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = tiny_setup(tmp)
            start = get_trainable_state(run.model)
            res = run.run()
            self.assertGreater(sum(float((res["state"][k] - start[k]).abs().sum()) for k in start), 0.0)
            self.assertEqual({r["language"] for r in run.final_test(res)}, {"mr", "en"})


class ClientBatches(unittest.TestCase):
    def test_batches_stay_within_client_and_repeat_exactly(self):
        t = fake_table("mr", 30, 3)
        rows = np.arange(0, 10)
        c = ClientData("mr_c0", "mr", t, rows)
        lens = set(t["ids"].map(len).to_numpy()[rows])
        for b in c.batches(0, 0, steps=5, batch_size=4, pad_id=1):
            self.assertEqual(b.input_ids.shape[0], 4)
            self.assertTrue(set(b.attention_mask.sum(1).numpy()) <= lens)
        a = [x.input_ids for x in c.batches(0, 3, 2, 4, 1)]
        b2 = [x.input_ids for x in c.batches(0, 3, 2, 4, 1)]
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b2)))


if __name__ == "__main__":
    unittest.main()
