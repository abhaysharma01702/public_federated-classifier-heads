"""Run one experiment to completion.

    python scripts/run_one.py --tok-dir <token tables> --runs-dir <out>/runs \
        --experiment E0 --method fedavg --setting iid12 --seed 0 [--set fl.rounds=20 --set model.head=linear]

* Idempotent: a run whose status.json says done is skipped.
* Resumes from its last checkpoint after a crash.
* An error marks the run failed (with the traceback) and exits with code 1.
Files in runs/<run_id>/: config.resolved.yaml, env.json, status.json, log.txt, metrics_round.csv,
timing.csv, comm.csv, final_test.csv, summary.json, best_state.pt, final_state.pt. Local-only runs save each client's best and final weights in client_states/. Local-only runs
keep no global model: they write local_clients.csv instead of metrics_round.csv and the state files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import yaml

import _bootstrap  # noqa: F401

from fedheads.utils.io import atomic_write_text, read_json, write_json


def parse_sets(items: list[str]) -> dict:
    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def env_record() -> dict:
    import peft
    import torch
    import transformers

    rec = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
           "peft": peft.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        here = Path(__file__).resolve().parent
        rec["git_commit"] = subprocess.run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()
    except Exception:
        rec["git_commit"] = None
    # On Ada the code is a `git archive` copy with no .git; ada/upload.sh writes VERSION instead.
    version = Path(__file__).resolve().parent.parent / "VERSION"
    if not rec["git_commit"] and version.exists():
        rec["git_commit"] = version.read_text(encoding="utf-8").split()[0]
    rec["host"] = platform.node()
    rec["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    return rec


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        atomic_write_text(path, pd.DataFrame(rows).to_csv(index=False))


def summarize(cfg: dict, final_rows: list[dict], best: dict | None) -> dict:
    df = pd.DataFrame(final_rows)
    out = {"run_id": cfg["run_id"], "best_round": best.get("round") if best else None,
           "best_validation_mean_f1": best.get("score") if best else None}
    if not df.empty:
        for state in ("best", "final"):
            d = df[df["state"] == state]
            if d.empty:
                continue
            out[f"test_{state}_mean_f1"] = float(d["macro_f1"].mean())
            out[f"test_{state}_worst_f1"] = float(d["macro_f1"].min())
            out[f"test_{state}_by_language"] = {r["language"]: float(r["macro_f1"]) for _, r in d.iterrows()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment")
    ap.add_argument("--tok-dir", required=True, type=Path)
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. fl.rounds=20")
    ap.add_argument("--local-files-only", action="store_true", help="laptop: use the cached backbone only")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from fedheads.runner.config import resolve

    cfg = resolve(args.experiment, args.method, args.setting, args.seed, parse_sets(args.set))
    run_dir = args.runs_dir / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.exists() and read_json(status_path).get("state") == "done":
        print(f"{cfg['run_id']}: already done, skipped")
        return 0
    log_file = open(run_dir / "log.txt", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t0 = time.time()
    write_json(status_path, {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_write_text(run_dir / "config.resolved.yaml", yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True))
    try:
        import torch

        from fedheads.data.loaders import load_token_tables
        from fedheads.eval.metrics import Evaluator
        from fedheads.fl.baselines import CentralizedRun, LocalOnlyRun
        from fedheads.fl.checkpoint import clear_checkpoints
        from fedheads.fl.server import FederatedRun
        from fedheads.models.model import build_model, count_parameters
        from fedheads.runner.setup import build_clients, eval_rows

        write_json(run_dir / "env.json", env_record())
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        use_amp = device.type == "cuda" and cfg["train"]["precision"] == "fp16"
        backbone = cfg["model"]["backbone"]
        tables = load_token_tables(args.tok_dir, backbone, cfg["data"]["languages"])
        if cfg["data"].get("drop_neutral"):  # two-class appendix
            from fedheads.data.loaders import drop_neutral
            tables = drop_neutral(tables)
        pad_id = read_json(args.tok_dir / "tokens_report.json")["backbones"][backbone]["pad_id"]
        model = build_model(cfg["model"], cfg["seed"], local_files_only=args.local_files_only).to(device)
        log(f"{cfg['run_id']} on {device}; parameters {count_parameters(model)}")
        clients = build_clients(tables, cfg)
        evaluator = Evaluator(tables, eval_rows(tables, cfg), pad_id, device, use_amp=use_amp,
                              n_classes=cfg["model"].get("n_classes", 3))
        log("clients: " + ", ".join(f"{c.client_id}={c.n}" for c in clients))
        mode = cfg["strategy"]["mode"]
        cls = {"federated": FederatedRun, "centralized": CentralizedRun, "local_only": LocalOnlyRun}[mode]
        runner = cls(cfg, model, clients, evaluator, run_dir, device, pad_id, log, use_amp)
        result = runner.run()
        final_rows = runner.final_test(result)
        if mode == "local_only":
            write_csv(run_dir / "timing.csv", result["timing"])
            write_csv(run_dir / "local_clients.csv",
                      [{"client": k, "language": v["rows"][0]["language"], "best_validation_f1": v["best_val"],
                        "best_step": v["best_step"]} for k, v in result["done"].items()])
            best = {"round": None, "score": LocalOnlyRun.validation_score(result)}
        else:
            logs = result["logs"]
            write_csv(run_dir / "metrics_round.csv", logs["metrics"])
            write_csv(run_dir / "timing.csv", logs["timing"])
            write_csv(run_dir / "comm.csv", logs["comm"])
            best = result["best"]
            if best["state"] is not None:
                torch.save(best["state"], run_dir / "best_state.pt")
            torch.save(runner.snapshot(result["state"]), run_dir / "final_state.pt")
        write_csv(run_dir / "final_test.csv", final_rows)
        summary = summarize(cfg, final_rows, {"round": best["round"], "score": best["score"]})
        summary["hours"] = round((time.time() - t0) / 3600, 4)
        write_json(run_dir / "summary.json", summary)
        clear_checkpoints(run_dir)
        write_json(status_path, {"state": "done", "hours": summary["hours"],
                                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"done in {summary['hours']:.3f} h: {json.dumps(summary)}")
        return 0
    except Exception:
        tb = traceback.format_exc()
        log("FAILED\n" + tb)
        write_json(status_path, {"state": "failed", "error": tb[-2000:], "hours": round((time.time() - t0) / 3600, 4)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
