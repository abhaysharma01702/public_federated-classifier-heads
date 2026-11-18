"""FedProx and FedAdam, copied from the earlier engine (tag fedblacx-before-move,
scriptfl/fl/algorithms.py) without SCAFFOLD.

Pure tensor functions, shared by the runner (fl/server.py, fl/client.py) and the tests.

* FedProx (Li et al., 2020): the proximal term mu/2 ||w - w_global||^2 adds mu (w - w_global)
  to each gradient, before clipping.
* FedAdam (Reddi et al., 2021): the weighted mean client update is a pseudo-gradient for a
  server Adam step, without bias correction, with tau inside the denominator.

Neither changes what a client sends, so bytes per round equal FedAvg's.
"""

from __future__ import annotations

import torch


def zeros_like_state(state: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def add_prox_grads(named_params: list, anchor: dict, mu: float) -> None:
    """FedProx: grad += mu (w - w_global), on the device of each parameter."""
    for name, p in named_params:
        if p.grad is not None:
            p.grad.add_(p.detach() - anchor[name], alpha=mu)


def server_adam_step(state: dict, mean_state: dict, opt: dict | None, cfg: dict) -> tuple[dict, dict]:
    """One FedAdam step. ``state``: current global trainable state; ``mean_state``: the
    weighted mean of the returned client states. Returns (new state, new optimizer state)."""
    b1, b2, tau, lr = cfg.get("beta1", 0.9), cfg.get("beta2", 0.99), cfg.get("tau", 1e-3), cfg["lr"]
    if opt is None:
        opt = {"m": zeros_like_state(state), "v": zeros_like_state(state)}
    m, v, new = {}, {}, {}
    for k, x in state.items():
        delta = mean_state[k] - x
        m[k] = b1 * opt["m"][k] + (1 - b1) * delta
        v[k] = b2 * opt["v"][k] + (1 - b2) * delta * delta
        new[k] = x + lr * m[k] / (v[k].sqrt() + tau)
    return new, {"m": m, "v": v}
