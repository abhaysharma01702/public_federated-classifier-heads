"""Reference baselines: centralized and local-only.

* Centralized: all clients' training rows pooled; one model and one optimizer that
  persists across rounds. Each "round" runs clients_per_round x K steps, so the total
  number of sequences matches FedAvg. Same evaluation cadence.
* Local-only: each client trains alone, with a persistent optimizer, for the steps it
  would get in FL on average: rounds x clients_per_round / N x K. Each client keeps its
  best state on its own language's validation set; test scores are averaged over that
  language's clients.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..data.loaders import Batch, pad, seeded_rng
from ..eval.metrics import macro_f1
from ..models.model import get_trainable_state, set_trainable_state
from .checkpoint import load_checkpoint, save_checkpoint
from .client import client_torch_seed, make_optimizer, make_scaler, train_steps
from .server import FederatedRun


class PooledData:
    """All clients' rows behind the ClientData batch interface."""

    def __init__(self, clients: list):
        self.clients = clients
        self.client_id = "pooled"
        self.language = "pooled"
        self.index = [(ci, r) for ci, c in enumerate(clients) for r in c.rows]

    @property
    def n(self) -> int:
        return len(self.index)

    def batches(self, seed, round_idx, steps, batch_size, pad_id):
        rng = seeded_rng("batches", seed, self.client_id, round_idx)
        need = steps * batch_size
        order = np.concatenate([rng.permutation(len(self.index)) for _ in range(-(-need // self.n))])[:need]
        for s in range(steps):
            picks = [self.index[i] for i in order[s * batch_size:(s + 1) * batch_size]]
            seqs = [self.clients[ci].table["ids"].iat[r] for ci, r in picks]
            labels = [int(self.clients[ci].table["label"].iat[r]) for ci, r in picks]
            ids, mask = pad(seqs, pad_id)
            yield Batch(ids, mask, torch.tensor(labels, dtype=torch.int64))


class CentralizedRun(FederatedRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooled = PooledData(self.clients)
        self.optimizer = make_optimizer(self.model, self.cfg["train"])

    def train_round(self, t, state):
        set_trainable_state(self.model, state)
        torch.manual_seed(client_torch_seed(self.cfg["seed"], "pooled", t))
        steps = self.cfg["fl"]["clients_per_round"] * self.cfg["train"]["local_steps"]
        batches = self.pooled.batches(self.cfg["seed"], t, steps, self.cfg["train"]["batch_sentences"], self.pad_id)
        st = train_steps(self.model, batches, self.optimizer, self.scaler, self.device, self.cfg["train"], self.use_amp)
        return get_trainable_state(self.model), st["sequences"], 0

    def extra_state(self):
        return {"optimizer": self.optimizer.state_dict()}

    def load_extra_state(self, extra):
        if extra and extra.get("optimizer"):
            self.optimizer.load_state_dict(extra["optimizer"])


class LocalOnlyRun:
    def __init__(self, cfg, model, clients, evaluator, run_dir, device, pad_id, logger, use_amp):
        self.cfg, self.model, self.clients, self.evaluator = cfg, model, clients, evaluator
        self.run_dir, self.device, self.pad_id, self.log, self.use_amp = run_dir, device, pad_id, logger, use_amp
        fl, tr = cfg["fl"], cfg["train"]
        self.total_steps = int(round(fl["rounds"] * fl["clients_per_round"] / len(clients) * tr["local_steps"]))
        self.n_checks = max(1, fl["rounds"] // fl["eval_every"])

    def _score(self, language: str, split: str) -> tuple[float, float, int]:
        y, p = self.evaluator.predict(self.model, language, self.evaluator.split_rows[split][language])
        return macro_f1(y, p, getattr(self.evaluator, "n_classes", 3)), float((y == p).mean()), int(len(y))

    def run(self) -> dict:
        init = get_trainable_state(self.model)
        ck = load_checkpoint(self.run_dir)
        done = ck["done"] if ck else {}
        timing = ck["timing"] if ck else []
        chunk = -(-self.total_steps // self.n_checks)
        for c in self.clients:
            if c.client_id in done:
                continue
            t0 = time.time()
            set_trainable_state(self.model, init)
            optimizer = make_optimizer(self.model, self.cfg["train"])
            scaler = make_scaler(self.device, self.use_amp)
            best = (-1.0, None, 0)
            seqs, steps_done = 0, 0
            for k in range(self.n_checks):
                steps = min(chunk, self.total_steps - steps_done)
                if steps <= 0:
                    break
                torch.manual_seed(client_torch_seed(self.cfg["seed"], c.client_id, k))
                batches = c.batches(self.cfg["seed"], k, steps, self.cfg["train"]["batch_sentences"], self.pad_id)
                st = train_steps(self.model, batches, optimizer, scaler, self.device, self.cfg["train"], self.use_amp)
                seqs += st["sequences"]
                steps_done += steps
                f1 = self._score(c.language, "validation")[0]
                if f1 > best[0]:
                    best = (f1, get_trainable_state(self.model), steps_done)
            final_state = get_trainable_state(self.model)
            # Every client's best and final weights are kept.
            wdir = Path(self.run_dir) / "client_states"
            wdir.mkdir(exist_ok=True)
            torch.save(best[1], wdir / f"{c.client_id}_best_state.pt")
            torch.save(final_state, wdir / f"{c.client_id}_final_state.pt")
            rows = []
            for name, stt in (("best", best[1]), ("final", final_state)):
                set_trainable_state(self.model, stt)
                f1, acc, n = self._score(c.language, "test")
                rows.append({"state": name, "client": c.client_id, "language": c.language,
                             "macro_f1": f1, "accuracy": acc, "n": n})
            done[c.client_id] = {"rows": rows, "best_val": best[0], "best_step": best[2]}
            secs = time.time() - t0
            timing.append({"round": len(done), "seconds": round(secs, 3), "sequences": seqs,
                           "sequences_per_second": round(seqs / secs, 2) if secs else None})
            save_checkpoint(self.run_dir, {"round": len(done), "done": done, "timing": timing})
            self.log(f"local-only {c.client_id}: best validation macro-F1 {best[0]:.4f}")
        return {"done": done, "timing": timing}

    @staticmethod
    def validation_score(result: dict) -> float:
        """Each client's best validation macro-F1 on its own language, averaged over the
        language's clients, then over languages."""
        per_lang: dict = {}
        for v in result["done"].values():
            per_lang.setdefault(v["rows"][0]["language"], []).append(v["best_val"])
        return float(np.mean([np.mean(x) for x in per_lang.values()]))

    @staticmethod
    def final_test(result: dict) -> list[dict]:
        """Average each language's clients."""
        import pandas as pd

        rows = pd.DataFrame([r for v in result["done"].values() for r in v["rows"]])
        agg = rows.groupby(["state", "language"]).agg(macro_f1=("macro_f1", "mean"), accuracy=("accuracy", "mean"),
                                                      n=("n", "first"), clients=("client", "count")).reset_index()
        agg.insert(1, "split", "test")
        return agg.to_dict("records")
