"""Run one experiment module.

    python scripts/run_module.py --module P0 --tok-dir <token tables> \
        --out-dir /kaggle/working/<module> --session-start <unix time of the first notebook cell>

The module file configs/modules/<module>.yaml lists lanes of runs. With two GPUs each lane
runs on its own GPU; with one, the lanes run one after the other.
* ``platform: ada`` in the module (one SLURM job, 6 h limit) uses the ``session_ada`` guard
  (5.5 h) instead of the Kaggle one (10.5 h); ``--session-start`` is then the job's start.
* A run already done in --out-dir is skipped without starting a process, so resubmitting a
  job continues where it stopped; an unfinished run resumes from its last checkpoint.
* Optional ``calibrate: true``: the speed benchmark runs first, and its measured speeds
  replace the planning numbers for the guard. An Ada module without it uses the speeds
  saved in configs/calibration_ada.json.
* Every run is a fresh subprocess (run_one.py), so GPU memory is released.
* The session guard skips any run that could cross the guard line; it stays pending for
  the next module. A failed run never stops the module.
* module_report.json lists every run: state, predicted and actual hours.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import _bootstrap  # noqa: F401

from fedheads.runner.config import resolve
from fedheads.runner.guard import may_start
from fedheads.runner.timing import predict_hours
from fedheads.utils.io import CONFIG_DIR, load_base_config, load_yaml, read_json, write_json

HERE = Path(__file__).resolve().parent


def platform_node() -> dict:
    """Where the module ran: host name, and the SLURM job ID on Ada."""
    import socket

    return {"host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}


def ada_calibration() -> dict | None:
    """Speeds measured on Ada (configs/calibration_ada.json), without the notes ("_" keys); None if absent."""
    path = CONFIG_DIR / "calibration_ada.json"
    if not path.exists():
        return None
    return {k: v for k, v in read_json(path).items() if not k.startswith("_")}


def run_args(spec: dict) -> list[str]:
    out = ["--experiment", spec["experiment"], "--method", spec["method"], "--setting", spec["setting"],
           "--seed", str(spec["seed"])]
    for s in spec.get("set", []):
        out += ["--set", s]
    return out


def overrides(spec: dict, extra_sets: list[str]) -> dict:
    """The same overrides run_one.py will apply, in the same order, so run IDs match."""
    from run_one import parse_sets

    return parse_sets(list(spec.get("set", [])) + list(extra_sets))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one experiment module")
    ap.add_argument("--module", required=True)
    ap.add_argument("--tok-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--session-start", type=float, default=None, help="unix time the Kaggle session or SLURM job started")
    ap.add_argument("--extra-set", action="append", default=[], help="override added to every run (tests)")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    start = args.session_start or time.time()
    module = load_yaml(CONFIG_DIR / "modules" / f"{args.module}.yaml")
    platform = module.get("platform", "kaggle")
    session = load_base_config()["session_ada" if platform == "ada" else "session"]
    out = args.out_dir
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    extra = ["--local-files-only"] if args.local_files_only else []
    for s in args.extra_set:
        extra += ["--set", s]

    import torch

    n_gpus = torch.cuda.device_count()
    report = {"module": args.module, "platform": platform, "node": platform_node(), "gpus": n_gpus, "runs": [], "calibration": None}

    # Every language a run needs must have token tables; a wrong input stops the module here,
    # before any GPU work.
    needed = set()
    for lane in module["lanes"]:
        for spec in lane:
            cfg = resolve(spec["experiment"], spec["method"], spec["setting"], spec["seed"],
                          overrides(spec, args.extra_set))
            needed |= {(cfg["model"]["backbone"], lang) for lang in cfg["data"]["languages"]}
    missing = sorted(f"{b}/{lang}.parquet" for b, lang in needed if not (args.tok_dir / b / f"{lang}.parquet").exists())
    if missing:
        report["summary"] = {"missing token tables": len(missing)}
        report["missing_token_tables"] = missing
        write_json(out / "module_report.json", report)
        print(f"=== Token tables missing in {args.tok_dir}: {missing}. Attach the tokenizing notebook's "
              "output that has them; no run was started.", flush=True)
        return 1
    calib = None
    if module.get("calibrate"):
        cal_path = out / "calibration.json"
        if not cal_path.exists():
            print("=== speed benchmark", flush=True)
            cmd = [sys.executable, str(HERE / "calibrate.py"), "--tok-dir", str(args.tok_dir), "--out", str(cal_path)] + extra
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"} if n_gpus else dict(os.environ)
            with open(out / "calibration.log", "w", encoding="utf-8") as lf:
                subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        if cal_path.exists():
            report["calibration"] = read_json(cal_path)
            calib = report["calibration"]["calibration"]
        else:
            # A broken environment must not go on to start long runs.
            tail = (out / "calibration.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            report["summary"] = {"calibration failed": 1}
            report["calibration_error"] = "\n".join(tail)
            write_json(out / "module_report.json", report)
            print("=== speed benchmark FAILED; no run was started. Last lines of calibration.log:\n" + "\n".join(tail), flush=True)
            return 1
    elif platform == "ada":
        # No benchmark in this module: use the speeds measured on Ada's RTX 2080 Ti (P0), the same
        # numbers make_modules.py planned with, instead of the slower planning speeds.
        calib = ada_calibration()
        report["calibration"] = {"source": "configs/calibration_ada.json", "calibration": calib}

    lanes = module["lanes"]
    if n_gpus < 2:  # one GPU or CPU: run the lanes one after the other
        lanes = [[spec for lane in lanes for spec in lane]]
    lock = threading.Lock()

    def run_lane(lane_idx: int, lane: list[dict]) -> None:
        env = dict(os.environ)
        if n_gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(lane_idx % max(n_gpus, 1))
        for spec in lane:
            cfg = resolve(spec["experiment"], spec["method"], spec["setting"], spec["seed"],
                          overrides(spec, args.extra_set))
            predicted = predict_hours(cfg, calib)
            elapsed = (time.time() - start) / 3600
            row = {"run_id": cfg["run_id"], "lane": lane_idx, "predicted_hours": round(predicted, 3),
                   "started_at_hours": round(elapsed, 3)}
            status_path = runs_dir / cfg["run_id"] / "status.json"
            earlier = read_json(status_path) if status_path.exists() else {}
            if earlier.get("state") == "done":
                row["state"] = "done"
                row["actual_hours"] = earlier.get("hours")
                row["skipped"] = "done earlier"
                print(f"=== lane {lane_idx}: {cfg['run_id']} done earlier; skipped", flush=True)
            elif not may_start(elapsed, predicted, session):
                row["state"] = "pending (session guard)"
            else:
                print(f"=== {time.strftime('%H:%M:%S')} lane {lane_idx}: {cfg['run_id']} (predicted {predicted:.2f} h)", flush=True)
                cmd = [sys.executable, str(HERE / "run_one.py"), "--tok-dir", str(args.tok_dir),
                       "--runs-dir", str(runs_dir)] + run_args(spec) + extra
                with open(out / f"lane{lane_idx}.log", "a", encoding="utf-8") as lf:
                    subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
                status = read_json(status_path) if status_path.exists() else {"state": "failed", "error": "no status"}
                row["state"] = status.get("state")
                row["actual_hours"] = status.get("hours")
                print(f"=== {time.strftime('%H:%M:%S')} lane {lane_idx}: {cfg['run_id']} -> {row['state']} ({row.get('actual_hours')} h)", flush=True)
            with lock:
                report["runs"].append(row)
                write_json(out / "module_report.json", report)

    threads = [threading.Thread(target=run_lane, args=(i, lane)) for i, lane in enumerate(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report["hours_total"] = round((time.time() - start) / 3600, 3)
    states = [r["state"] for r in report["runs"]]
    report["summary"] = {s: states.count(s) for s in set(states)}
    write_json(out / "module_report.json", report)
    print(f"=== module {args.module} finished: {report['summary']} in {report['hours_total']} h", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
