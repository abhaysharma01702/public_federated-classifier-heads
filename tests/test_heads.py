"""The six heads and the <s> side check: shapes, padding, sizes, bytes."""

import unittest

import torch

import _path  # noqa: F401

from fedheads.models.heads import HEADS, MAIN_HEADS, build_head, matched_mlp_width

# Parameter counts at XLM-R base's hidden size 768, 3 classes (proposal of 9 November 2026).
EXPECTED_768 = {"linear": 2_307, "linear_cls": 2_307, "cnn": 295_427, "bilstm": 2_102_787,
                "cnn_bilstm": 2_398_211, "mlp": 2_397_835, "attention": 100_867}
LORA_768 = 294_912   # r 8 on query and value, 12 layers: 12 x 2 x (768 x 8 + 8 x 768)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class Sizes(unittest.TestCase):
    def test_parameter_counts_at_768(self):
        for name, want in EXPECTED_768.items():
            with self.subTest(head=name):
                self.assertEqual(n_params(build_head(name, 768)), want)

    def test_mlp_matches_cnn_bilstm(self):
        self.assertEqual(matched_mlp_width(768), 3_106)
        for h in (16, 64, 768):
            a, b = n_params(build_head("mlp", h)), n_params(build_head("cnn_bilstm", h))
            self.assertLess(abs(a - b) / b, 0.001, h)

    def test_bytes_per_round(self):
        # LoRA + head, fp32, one client: what FedAvg sends each way per round.
        mb = {k: (LORA_768 + v) * 4 / 1e6 for k, v in EXPECTED_768.items()}
        self.assertAlmostEqual(mb["linear"], 1.188876, places=6)   # = 14,266,512 / 12 in job 66745's comm.csv
        self.assertAlmostEqual(mb["cnn_bilstm"], 10.772492, places=6)


class Behaviour(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.h = 16
        self.lengths = [7, 4, 2]
        self.hidden = torch.randn(3, 7, self.h)
        self.mask = torch.tensor([[1] * n + [0] * (7 - n) for n in self.lengths])

    def test_shapes(self):
        for name in HEADS:
            with self.subTest(head=name):
                head = build_head(name, self.h).eval()
                logits, pooled = head(self.hidden, self.mask)
                self.assertEqual(tuple(logits.shape), (3, 3))
                self.assertEqual(pooled.shape[0], 3)
                self.assertTrue(torch.isfinite(logits).all())

    def test_padding_never_changes_the_output(self):
        # Same sentences, padded to 12 with random garbage in the padding positions.
        extra = torch.randn(3, 5, self.h) * 100
        hidden2 = torch.cat([self.hidden, extra], dim=1)
        for i, n in enumerate(self.lengths):
            hidden2[i, n:7] = torch.randn(7 - n, self.h) * 100
        mask2 = torch.tensor([[1] * n + [0] * (12 - n) for n in self.lengths])
        for name in HEADS:
            with self.subTest(head=name):
                head = build_head(name, self.h).eval()
                a, _ = head(self.hidden, self.mask)
                b, _ = head(hidden2, mask2)
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    def test_single_sentence_equals_batched(self):
        # A sentence alone (no padding) scores the same as inside a padded batch.
        for name in HEADS:
            with self.subTest(head=name):
                head = build_head(name, self.h).eval()
                batched, _ = head(self.hidden, self.mask)
                n = self.lengths[2]
                alone, _ = head(self.hidden[2:3, :n], self.mask[2:3, :n])
                torch.testing.assert_close(batched[2:3], alone, rtol=1e-5, atol=1e-5)

    def test_attention_ignores_padding(self):
        head = build_head("attention", self.h)
        w = head.weights(self.hidden, self.mask)
        self.assertTrue((w[self.mask == 0] == 0).all())
        torch.testing.assert_close(w.sum(dim=1), torch.ones(3))

    def test_gradients_reach_every_parameter(self):
        for name in HEADS:
            with self.subTest(head=name):
                head = build_head(name, self.h).train()
                logits, _ = head(self.hidden, self.mask)
                logits.sum().backward()
                for pname, p in head.named_parameters():
                    self.assertIsNotNone(p.grad, pname)
                    self.assertGreater(float(p.grad.abs().sum()), 0.0, pname)

    def test_main_heads_are_the_six_heads(self):
        self.assertEqual(sorted(MAIN_HEADS), sorted(["linear", "cnn", "bilstm", "cnn_bilstm", "mlp", "attention"]))


if __name__ == "__main__":
    unittest.main()
