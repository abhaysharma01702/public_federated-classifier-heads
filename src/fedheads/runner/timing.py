"""Predicted run time.

Hours = training sequences / training speed + evaluated sequences / evaluation speed + loading.

* Before any calibration, the planning numbers below are used (the earlier engine's: 150 training
  and 500 evaluation sequences per second on one T4).
* After a module with ``calibrate: true``, the measured speeds replace them. The calibration
  measures each head's training speed (``train_seq_per_s_by_head``), so a heavier head is
  predicted to take longer; a head without a measured speed uses the linear head's.
* Record predicted against actual in module_report.json and refit when they drift.
"""

from __future__ import annotations

PLANNING = {"train_seq_per_s": 150.0, "eval_seq_per_s": 500.0, "load_seconds": 120.0}

# Official split sizes before cleaning; replaced by the real counts once the
# token tables exist (calibration writes validation_rows and test_rows).
VALIDATION_ROWS = {"mr": 6000, "en": 2000, "bn": 1567, "hi": 833}
TEST_ROWS = {"mr": 6000, "en": 12284, "bn": 1586, "hi": 833}


def train_speed(calib: dict, head: str) -> float:
    return float((calib.get("train_seq_per_s_by_head") or {}).get(head, calib["train_seq_per_s"]))


def predict_hours(cfg: dict, calib: dict | None = None) -> float:
    c = {**PLANNING, **(calib or {})}
    fl, tr = cfg["fl"], cfg["train"]
    langs = cfg["data"]["languages"]
    val_rows = {**VALIDATION_ROWS, **(c.get("validation_rows") or {})}
    test_rows = {**TEST_ROWS, **(c.get("test_rows") or {})}
    sentences = fl["rounds"] * fl["clients_per_round"] * tr["local_steps"] * tr["batch_sentences"]
    train_s = sentences / train_speed(c, cfg["model"]["head"])
    evals = max(1, fl["rounds"] // fl["eval_every"])
    mode = cfg["strategy"]["mode"]
    if mode == "local_only":   # every client checks its own language at each evaluation
        clients = cfg["setting_cfg"]["clients"]
        eval_seqs = sum(int(clients[lang]) * val_rows[lang] for lang in langs) * evals
        test_seqs = 2 * sum(int(clients[lang]) * test_rows[lang] for lang in langs)
    else:
        eval_seqs = evals * sum(val_rows[lang] for lang in langs)
        test_seqs = 2 * sum(test_rows[lang] for lang in langs)   # best and final state
    eval_s = (eval_seqs + test_seqs) / c["eval_seq_per_s"]
    return (train_s + eval_s + c["load_seconds"]) / 3600
