"""The federated round loop.

* The client-sampling schedule for all rounds is drawn up front from the run seed.
* Aggregation: weighted by each participating client's training-set size, over the
  trainable tensors only (LoRA and head).
* Every ``eval_every`` rounds: validation; the best state by the mean validation
  macro-F1 over languages is kept.
* Every ``checkpoint_every`` rounds: a checkpoint, used only to recover from a crash.
  Resuming continues the same schedule and the same batches.
* At the end: test evaluation of the best and the final state.
* Communication: bytes sent up and down per round, for the cost analysis (experiment 6).
* From the strategy config: ``prox_mu`` > 0 gives FedProx; ``server: {name: adam, lr, ...}``
  gives FedAdam. Its moments are saved in checkpoints, so resume is exact.
"""

from __future__ import annotations

import time

import numpy as np

from ..models.model import get_trainable_state, set_trainable_state
from .algorithms import add_prox_grads, server_adam_step
from .checkpoint import load_checkpoint, save_checkpoint
from .client import local_update, make_scaler


def sampling_schedule(n_clients: int, per_round: int, rounds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, 7919])
    return np.stack([np.sort(rng.choice(n_clients, size=min(per_round, n_clients), replace=False))
                     for _ in range(rounds)])


def weighted_average(states: list[dict], weights: list[float]) -> dict:
    total = float(sum(weights))
    out = {}
    for key in states[0]:
        out[key] = sum(s[key] * (w / total) for s, w in zip(states, weights))
    return out


def state_bytes(state: dict) -> int:
    return int(sum(t.numel() * 4 for t in state.values()))  # sent as fp32


class FederatedRun:
    def __init__(self, cfg: dict, model, clients: list, evaluator, run_dir, device, pad_id: int, logger, use_amp: bool):
        self.cfg, self.model, self.clients, self.evaluator = cfg, model, clients, evaluator
        self.run_dir, self.device, self.pad_id, self.log, self.use_amp = run_dir, device, pad_id, logger, use_amp
        fl = cfg["fl"]
        self.rounds = fl["rounds"]
        self.schedule = sampling_schedule(len(clients), fl["clients_per_round"], self.rounds, cfg["seed"])
        self.scaler = make_scaler(device, use_amp)
        st = cfg.get("strategy") or {}
        self.prox_mu = float(st.get("prox_mu", 0.0))
        self.server_cfg = st.get("server") or {"name": "average"}
        if self.server_cfg["name"] not in ("average", "adam"):
            raise ValueError(f"unknown server optimizer {self.server_cfg['name']}")
        self.server_opt = None   # FedAdam moments

    def run(self, stop_after_round: int | None = None) -> dict:
        """Train to the end (or stop after ``stop_after_round``, for the crash test)."""
        ck = load_checkpoint(self.run_dir)
        if ck is None:
            state = get_trainable_state(self.model)
            start, best = 0, {"score": -1.0, "round": None, "state": None}
            logs = {"metrics": [], "timing": [], "comm": []}
        else:
            state, start, best, logs = ck["state"], ck["round"], ck["best"], ck["logs"]
            if ck.get("scaler"):
                self.scaler.load_state_dict(ck["scaler"])
            self.load_extra_state(ck.get("extra"))
            if not np.array_equal(ck["schedule"], self.schedule):
                raise RuntimeError("checkpoint was made with a different sampling schedule")
            self.log(f"resumed from round {start}")
        fl = self.cfg["fl"]
        for t in range(start, self.rounds):
            t0 = time.time()
            state, seqs, n_comm = self.train_round(t, state)
            secs = time.time() - t0
            logs["timing"].append({"round": t + 1, "seconds": round(secs, 3), "sequences": seqs,
                                   "sequences_per_second": round(seqs / secs, 2) if secs else None})
            b = state_bytes(state)
            logs["comm"].append({"round": t + 1, "bytes_up": b * n_comm, "bytes_down": b * n_comm})
            if (t + 1) % fl["eval_every"] == 0 or t + 1 == self.rounds:
                rows = self.evaluate_snapshot(self.snapshot(state), "validation")
                logs["metrics"] += [{"round": t + 1, **r} for r in rows]
                score = self.evaluator.score(rows)
                if score > best["score"]:
                    best = {"score": score, "round": t + 1, "state": self.snapshot(state)}
                self.log(f"round {t + 1}: mean validation macro-F1 {score:.4f} (best {best['score']:.4f} at {best['round']})")
            if (t + 1) % fl["checkpoint_every"] == 0 and t + 1 < self.rounds:
                save_checkpoint(self.run_dir, {"state": state, "round": t + 1, "best": best, "logs": logs,
                                               "schedule": self.schedule, "scaler": self.scaler.state_dict(),
                                               "extra": self.extra_state()})
            if stop_after_round is not None and t + 1 >= stop_after_round:
                return {"stopped": True, "state": state, "logs": logs, "best": best}
        return {"stopped": False, "state": state, "logs": logs, "best": best}

    def train_round(self, t: int, state: dict) -> tuple[dict, int, int]:
        """One round. Returns (new global state, sequences processed, clients communicating)."""
        picked = [self.clients[i] for i in self.schedule[t]]
        states, weights, seqs = [], [], 0
        for c in picked:
            hook = None
            if self.prox_mu:
                anchor = {k: v.to(self.device) for k, v in state.items()}
                hook = lambda named, a=anchor, mu=self.prox_mu: add_prox_grads(named, a, mu)  # noqa: E731
            s, st = local_update(self.model, state, c, t, self.cfg, self.device, self.scaler, self.pad_id,
                                 self.use_amp, before_clip=hook)
            states.append(s)
            weights.append(c.n)
            seqs += st["sequences"]
        return self.server_step(state, weighted_average(states, weights)), seqs, len(picked)

    def server_step(self, state: dict, mean_state: dict) -> dict:
        if self.server_cfg["name"] == "adam":
            new, self.server_opt = server_adam_step(state, mean_state, self.server_opt, self.server_cfg)
            return new
        return mean_state

    # ----- state for checkpoints and evaluation -------------------------------------------------
    def extra_state(self) -> dict | None:
        return {"server_opt": self.server_opt} if self.server_opt is not None else None

    def load_extra_state(self, extra: dict | None) -> None:
        if extra:
            self.server_opt = extra.get("server_opt")

    def snapshot(self, state: dict):
        """The best and final models: here the global trainable state."""
        return {k: v.clone() for k, v in state.items()}

    def evaluate_snapshot(self, snap, split: str) -> list[dict]:
        set_trainable_state(self.model, snap)
        return self.evaluator.evaluate(self.model, split)

    def final_test(self, result: dict) -> list[dict]:
        rows = []
        for name, snap in (("best", result["best"]["state"]), ("final", self.snapshot(result["state"]))):
            if snap is None:
                continue
            rows += [{"state": name, **r} for r in self.evaluate_snapshot(snap, "test")]
        return rows
