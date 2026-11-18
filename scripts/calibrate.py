"""Speed benchmark: measure real speed on the GPU.

    python scripts/calibrate.py --tok-dir <token tables> --out <dir>/calibration.json

Measures:
* training sequences per second at batch 16, for every registered head;
* evaluation sequences per second;
* seconds per round for 5 rounds of FedAvg on the main setting;
* a resume-after-kill check on the GPU (4 rounds straight against 2 + resume + 2): the largest
  difference in the trained parameters, which on a GPU is small but not zero;
* peak GPU memory.
The predicted hours of each run follow from these numbers (runner/timing.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import _bootstrap  # noqa: F401

from fedheads.utils.io import read_json, write_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Speed benchmark")
    ap.add_argument("--tok-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--setting", default="iid12")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. model.backbones.xlmr=<path>")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    import torch

    from run_one import parse_sets
    from fedheads.data.loaders import load_token_tables
    from fedheads.eval.metrics import Evaluator
    from fedheads.fl.client import make_optimizer, make_scaler, train_steps
    from fedheads.fl.server import FederatedRun
    from fedheads.models.heads import HEADS
    from fedheads.models.model import build_model, get_trainable_state, set_trainable_state
    from fedheads.runner.config import resolve
    from fedheads.runner.setup import build_clients, eval_rows
    from fedheads.runner.timing import predict_hours

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    over = parse_sets(args.set)
    cfg = resolve("E0", "fedavg", args.setting, 0, over)
    backbone = cfg["model"]["backbone"]
    tables = load_token_tables(args.tok_dir, backbone, cfg["data"]["languages"])
    pad_id = read_json(args.tok_dir / "tokens_report.json")["backbones"][backbone]["pad_id"]
    out: dict = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "steps": args.steps}

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 1. Training throughput per head, on the language with the longest sentences.
    clients = build_clients(tables, cfg)
    longest = max(cfg["data"]["languages"], key=lambda lang: tables[lang]["ids"].map(len).mean())
    client = next(c for c in clients if c.language == longest)
    out["benchmark_client"] = client.client_id
    by_head = {}
    for head in sorted(HEADS):
        model = build_model({**cfg["model"], "head": head}, 0, local_files_only=args.local_files_only).to(device)
        opt, scaler = make_optimizer(model, cfg["train"]), make_scaler(device, use_amp)
        train_steps(model, client.batches(0, 999, 5, 16, pad_id), opt, scaler, device, cfg["train"], use_amp)
        sync()
        t0 = time.time()
        st = train_steps(model, client.batches(0, 1000, args.steps, 16, pad_id), opt, scaler, device,
                         cfg["train"], use_amp)
        sync()
        by_head[head] = round(st["sequences"] / (time.time() - t0), 2)
        print(head, by_head[head], "training sequences/s", flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2. Evaluation throughput (validation, every language), with the default head.
    model = build_model(cfg["model"], 0, local_files_only=args.local_files_only).to(device)
    rows = eval_rows(tables, cfg)
    ev = Evaluator(tables, rows, pad_id, device, use_amp=use_amp)
    sync()
    t0 = time.time()
    res_rows = ev.evaluate(model, "validation")
    sync()
    n = sum(r["n"] for r in res_rows)
    out["eval_seq_per_s"] = round(n / (time.time() - t0), 2)
    print("eval", out["eval_seq_per_s"], "sequences/s", flush=True)

    # 3. Seconds per round, FedAvg (no evaluation, no checkpoint).
    init = get_trainable_state(model)
    rcfg = resolve("E0", "fedavg", args.setting, 0, {**over, "fl": {"rounds": 5, "eval_every": 1000, "checkpoint_every": 1000}})
    with tempfile.TemporaryDirectory() as tmp:
        res = FederatedRun(rcfg, model, build_clients(tables, rcfg), ev, tmp, device, pad_id, lambda m: None, use_amp).run()
    secs = [r["seconds"] for r in res["logs"]["timing"]]
    out["seconds_per_round_fedavg"] = round(sum(secs[1:]) / max(len(secs) - 1, 1), 2)
    print("fedavg", out["seconds_per_round_fedavg"], "s/round", flush=True)

    # 4. Resume after a kill, on the GPU.
    rcfg = resolve("E0", "fedavg", args.setting, 0, {**over, "fl": {"rounds": 4, "eval_every": 1000, "checkpoint_every": 2}})
    finals = []
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        for run_dir, stop in ((a, None), (b, 2), (b, None)):
            set_trainable_state(model, init)
            r = FederatedRun(rcfg, model, build_clients(tables, rcfg), ev, run_dir, device, pad_id, lambda m: None, use_amp)
            res = r.run(stop_after_round=stop)
            if stop is None:
                finals.append(res["state"])
    out["resume_max_abs_diff"] = float(max((finals[0][k] - finals[1][k]).abs().max() for k in finals[0]))
    print("resume max abs diff", out["resume_max_abs_diff"], flush=True)

    if device.type == "cuda":
        out["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    calib = {"train_seq_per_s": by_head.get("linear", min(by_head.values())), "train_seq_per_s_by_head": by_head,
             "eval_seq_per_s": out["eval_seq_per_s"], "load_seconds": 60.0,
             "validation_rows": {k: int(len(v)) for k, v in rows["validation"].items()},
             "test_rows": {k: int(len(v)) for k, v in rows["test"].items()}}
    for method in ("fedavg", "centralized", "local_only"):
        for head in sorted(HEADS):
            mcfg = resolve("E0", method, args.setting, 0, {**over, "model": {"head": head}})
            out[f"predicted_hours_100_rounds_{method}_{head}"] = round(predict_hours(mcfg, calib), 3)
    out["calibration"] = calib
    write_json(args.out, out)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
