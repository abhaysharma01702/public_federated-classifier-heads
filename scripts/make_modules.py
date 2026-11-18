"""Write the module files configs/modules/<M>.yaml.

    python scripts/make_modules.py           # write every module in MODULES
    python scripts/make_modules.py --check   # exit 1 if a written file differs

Module files are written by this script, never by hand. Runs are listed most important first and
each goes to the lane with the shorter predicted time so far, so the session guard (which skips
runs at the end of a lane) only ever drops the least important, and both GPUs finish at about the
same time. A module whose longer lane plus setup is predicted above 10 hours is refused.
Ada modules (``platform: ada``; one GPU per SLURM job, 6 h limit) have a single lane and are
refused above 5 hours including setup, which keeps them under the 5.5 h job guard.

Modules so far:
* TEST: a tiny module for the local test of run_module.py; never run on Kaggle.
* ADA_TEST: the same tiny runs as the first Ada job's check, after tokenizing.
* P0: the pilot (experiment 0) with the benchmark of every head.
* T1, T2, T3: tuning, the head learning rate for each of the six heads.
The full runs are added once configs/tuned.yaml is written after tuning.
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from run_one import parse_sets
from fedheads.runner.config import resolve
from fedheads.runner.timing import predict_hours
from fedheads.utils.io import CONFIG_DIR, load_yaml

# Measured speeds replace PLANNING in runner/timing.py once a benchmark has run on Kaggle.
CALIB_PATH = CONFIG_DIR / "calibration_ada.json"   # measured on Ada (P0); planning speeds if absent
CALIB = {k: v for k, v in json.loads(CALIB_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")} \
    if CALIB_PATH.exists() else None
SETUP_HOURS = 0.4          # installing and cloning plus the benchmark
MAX_MODULE_HOURS = 10.0
ADA_SETUP_HOURS = 0.3      # GPU check, data check and copy to scratch, benchmark
ADA_MAX_HOURS = 5.0

TINY = ["fl.rounds=2", "fl.eval_every=2", "train.local_steps=1"]

# Tuning grid. Written with a decimal point so YAML reads them as numbers, not text.
# Middle values first, so the guard could only ever drop an edge value.
TUNE_LR_HEAD = ["1.0e-3", "3.0e-3", "3.0e-4", "1.0e-2"]
TUNE_MODULES = {"T1": ["linear", "cnn"], "T2": ["mlp", "attention"], "T3": ["bilstm", "cnn_bilstm"]}
# Edge rule: heads whose best rate was the lowest grid value (3e-4) get one run at 1e-4.
# Modules already run on Ada: kept as they ran, even where refitted speeds now predict them
# above the limit.
RAN_ALREADY = {"P0", "T1", "T2", "T3"}
FA_SERVER_LR = ["3.0e-3", "1.0e-2", "3.0e-2"]   # FedAdam server rate grid
E2_SEEDS = [0, 1, 2]
E2_METHODS = ["fedavg", "centralized", "local_only"]
MAIN_HEADS_ORDER = ["linear", "cnn", "mlp", "attention", "bilstm", "cnn_bilstm"]
E5_HEADS = ["linear", "cnn_bilstm", "mlp"]   # lightest, heaviest, and its parameter-matched control
E2_PACK_HOURS = 4.6       # predicted run hours per Ada module, below ADA_MAX_HOURS minus setup
TUNE_EDGE = {"T4": [("cnn", "1.0e-4"), ("bilstm", "1.0e-4"), ("cnn_bilstm", "1.0e-4")]}


def spec(experiment: str, method: str, setting: str, seed: int, sets: list[str] | None = None) -> dict:
    return {"experiment": experiment, "method": method, "setting": setting, "seed": seed, "set": list(sets or [])}


def tuned_sets(tuned: dict, head: str, method: str) -> list[str]:
    """The tuned learning rate per head (configs/tuned.yaml, written after tuning)."""
    return list((tuned.get(head) or {}).get(method, (tuned.get(head) or {}).get("all", [])))


def hours(s: dict) -> float:
    return predict_hours(resolve(s["experiment"], s["method"], s["setting"], s["seed"], parse_sets(s["set"])), CALIB)


def two_lanes(runs: list[dict]) -> list[list[dict]]:
    lanes, load = [[], []], [0.0, 0.0]
    for s in runs:
        k = 0 if load[0] <= load[1] else 1
        lanes[k].append(s)
        load[k] += hours(s)
    return lanes


def plan(tuned: dict) -> dict[str, tuple]:
    """{module: (description, calibrate, runs most important first[, platform])}."""
    out = {}
    out["TEST"] = ("Tiny module for the local test of run_module.py; never run on Kaggle", True,
                   [spec("T", "fedavg", "iid12", 0, TINY), spec("T", "local_only", "iid12", 0, TINY),
                    spec("T", "centralized", "iid12", 0, TINY)])
    out["ADA_TEST"] = ("Tiny check runs for the first Ada job, after tokenizing", True,
                       [spec("T", "fedavg", "iid12", 0, TINY), spec("T", "local_only", "iid12", 0, TINY),
                        spec("T", "centralized", "iid12", 0, TINY)], "ada")
    # Experiment 0, the pilot: linear and CNN-BiLSTM, FedAvg, seed 0, 20 rounds.
    # The benchmark first measures every head's speed (calibrate), for planning the later modules.
    out["P0"] = ("Pilot: linear and CNN-BiLSTM, FedAvg, seed 0, 20 rounds; benchmark of every head", True,
                 [spec("E0", "fedavg", "iid12", 0, ["model.head=linear", "fl.rounds=20"]),
                  spec("E0", "fedavg", "iid12", 0, ["model.head=cnn_bilstm", "fl.rounds=20"])], "ada")
    # Experiment 1, tuning: head learning rate, FedAvg, seed 0, 100 rounds, chosen on validation.
    for name, heads in TUNE_MODULES.items():
        out[name] = (f"Tuning: head learning rate for {' and '.join(heads)}, FedAvg, seed 0, 100 rounds", False,
                     [spec("E1", "fedavg", "iid12", 0, [f"model.head={h}", f"train.lr_head={lr}"])
                      for lr in TUNE_LR_HEAD for h in heads], "ada")
    for name, pairs in TUNE_EDGE.items():
        out[name] = ("Tuning edge runs: " + ", ".join(f"{h} at {lr}" for h, lr in pairs) + "; FedAvg, seed 0, 100 rounds", False,
                     [spec("E1", "fedavg", "iid12", 0, [f"model.head={h}", f"train.lr_head={lr}"]) for h, lr in pairs], "ada")
    # Experiment 2, the main comparison, once configs/tuned.yaml exists.
    # Seed 0 of every head and method first, then seeds 1 and 2; packed in order into Ada modules.
    if tuned:
        runs = [spec("E2", m, "iid12", seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, m))
                for seed in E2_SEEDS for m in E2_METHODS for h in MAIN_HEADS_ORDER]
        for i, chunk in enumerate(pack(runs, E2_PACK_HOURS), 1):
            out[f"E2_{i:02d}"] = (f"Experiment 2 part {i}: main comparison, 200 rounds, tuned head learning rates", False, chunk, "ada")
        # 17 November (user): re-run seed 0 local-only on Ada to save per-client models (E2_04, E2_05 ran
        # before per-client saving, code 7b8ab8a); same runs, new output folder; scores compared with the originals.
        e2l = [s for s in runs if s["method"] == "local_only" and s["seed"] == 0]
        for i, chunk in enumerate(pack(e2l, E2_PACK_HOURS), 1):
            if i == 1:
                out["E2L_01"] = ("Experiment 2, seed 0 local-only re-run part 1, to save per-client models", False, chunk, "ada")
                # 17 November (user): E2L_01 stopped after 3 runs so Ada can take the FedAdam check; the 4th on Kaggle.
                out["E2L_K3"] = ("Experiment 2, seed 0 local-only re-run, last run of part 1 (Kaggle)", True, chunk[3:], "kaggle")
            else:   # 17 November (user): part 2 on Kaggle, so Ada is free for the FedAdam check
                out[f"E2L_K{i}"] = (f"Experiment 2, seed 0 local-only re-run part {i} (Kaggle), to save per-client models", True, chunk, "kaggle")
        # Experiment 3: label skew. FedAvg at alpha 0.5 and 0.1, FedProx at 0.1,
        # then FedAdam at 0.1 once its server rate is tuned (FA_TUNE: linear head, seed 0, 100 rounds).
        out["FA_TUNE"] = ("FedAdam server learning rate, linear head, alpha 0.1, seed 0, 100 rounds", False,
                          [spec("E3", "fedadam", "dir01", 0, ["model.head=linear", f"strategy.server.lr={lr}"]
                                + tuned_sets(tuned, "linear", "fedavg")) for lr in FA_SERVER_LR], "ada")
        e3 = [spec("E3", m, st, seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
              for seed in E2_SEEDS for m, st in (("fedavg", "dir01"), ("fedprox", "dir01"), ("fedavg", "dir05"))
              for h in MAIN_HEADS_ORDER]
        fa_lr = (tuned.get("fedadam") or {}).get("server_lr")
        for i, chunk in enumerate(pack(e3, E2_PACK_HOURS), 1):
            out[f"E3_{i:02d}"] = (f"Experiment 3 part {i}: label skew (FedAvg at alpha 0.1 and 0.5, FedProx at 0.1), 200 rounds", False, chunk, "ada")
        if fa_lr:   # separate modules, so adding them never renumbers E3_*
            e3a = [spec("E3", "fedadam", "dir01", seed, [f"model.head={h}", "fl.rounds=200", f"strategy.server.lr={fa_lr}"]
                        + tuned_sets(tuned, h, "fedavg")) for seed in E2_SEEDS for h in MAIN_HEADS_ORDER]
            # 17 November (user, to finish sooner): parts 1 to 3 stay on Ada; parts 4 and 5 run on Kaggle as one module.
            rest = []
            for i, chunk in enumerate(pack(e3a, E2_PACK_HOURS), 1):
                if i <= 3:
                    out[f"E3A_{i:02d}"] = (f"Experiment 3, FedAdam part {i}: alpha 0.1, server rate {fa_lr}, 200 rounds", False, chunk, "ada")
                else:
                    rest += chunk
            if rest:
                out["E3A_K1"] = (f"Experiment 3, FedAdam (Kaggle): alpha 0.1, server rate {fa_lr}, 200 rounds", True, rest, "kaggle")
            # (17 November, user): seed-2 linear and CNN collapsed on Kaggle; the same two runs on Ada.
            chk = [s for s in e3a if s["seed"] == 2 and s["set"][0] in ("model.head=linear", "model.head=cnn")]
            out["E3A_CHK"] = ("FedAdam seed-2 linear and CNN on Ada: check of the collapse on Kaggle", False, chk, "ada")
        # Experiment 4: Marathi clients by domain, FedAvg, 6 heads x 3 seeds, 12 clients.
        e4 = [spec("E4", "fedavg", "dom12", seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
              for seed in E2_SEEDS for h in MAIN_HEADS_ORDER]
        # 17 November (user, to finish sooner): all of experiment 4 on Kaggle, four notebooks in parallel.
        for i, (a, b) in enumerate(((0, 5), (5, 10), (10, 14), (14, 18)), 1):
            out[f"E4_K{i}"] = (f"Experiment 4 (Kaggle) part {i}: Marathi clients by domain, FedAvg, 200 rounds", True, e4[a:b], "kaggle")
        # Experiment 5: 6 and 24 clients, all clients every round, FedAvg, three heads x 3 seeds.
        # (amended): the 24-client runs go to Kaggle (E5_K1), next to partial participation;
        # the 6-client runs stay on Ada.
        e5 = [spec("E5", "fedavg", "iid6", seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
              for seed in E2_SEEDS for h in E5_HEADS]
        # (amended again, 16 November): the 6-client runs also go to Kaggle (spare quota).
        out["E5_K3"] = ("Experiment 5 (Kaggle) part 3: 6 clients, FedAvg, 200 rounds", True, e5, "kaggle")
        e5k = [spec("E5", "fedavg", "iid24", seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
               for seed in E2_SEEDS for h in E5_HEADS]
        out["E5_K1"] = ("Experiment 5 (Kaggle) part 1: 24 clients, all every round, FedAvg, seeds 0 and 1, 200 rounds", True, e5k[:6], "kaggle")
        out["E5_K2"] = ("Experiment 5 (Kaggle) part 2: 24 clients, all every round, FedAvg, seed 2, 200 rounds", True, e5k[6:], "kaggle")
        # Experiment 5, partial participation: 24 clients, 12 sampled per round; same heads and seeds.
        e5p = [spec("E5", "fedavg", "iid24p", seed, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
               for seed in E2_SEEDS for h in E5_HEADS]
        # run on Kaggle (T4 x2, two lanes, one session), speeds measured first (calibrate).
        out["E5P_K1"] = ("Experiment 5, partial participation (Kaggle): 24 clients, 12 per round, FedAvg, 200 rounds", True, e5p, "kaggle")
        # Experiment 8: two classes, neutral removed; FedAvg, seed 0, tuned rates reused.
        e8 = [spec("E8", "fedavg", "iid12_2c", 0, [f"model.head={h}", "fl.rounds=200"] + tuned_sets(tuned, h, "fedavg"))
              for h in MAIN_HEADS_ORDER]
        for i, chunk in enumerate(pack(e8, E2_PACK_HOURS), 1):
            out[f"E8_{i:02d}"] = (f"Experiment 8 part {i}: two classes without neutral, FedAvg, seed 0, 200 rounds", False, chunk, "ada")
    return out


def pack(runs: list[dict], limit: float) -> list[list[dict]]:
    """Split runs, in order, into consecutive groups of at most `limit` predicted hours."""
    out, cur, load = [], [], 0.0
    for s in runs:
        h = hours(s)
        if cur and load + h > limit:
            out.append(cur); cur, load = [], 0.0
        cur.append(s); load += h
    if cur:
        out.append(cur)
    return out


def flow(item: str) -> str:
    """A YAML flow item; one holding a list or a comma is quoted."""
    return json.dumps(item) if any(c in item for c in "[],") else item


def render(name: str, what: str, calibrate: bool, runs: list[dict], platform: str = "kaggle") -> str:
    if platform == "ada":
        lanes, setup, limit = [list(runs)], ADA_SETUP_HOURS, ADA_MAX_HOURS
    else:
        lanes, setup, limit = two_lanes(runs), SETUP_HOURS, MAX_MODULE_HOURS
    lane_h = [sum(hours(s) for s in lane) for lane in lanes]
    if max(lane_h) + setup > limit and name not in RAN_ALREADY:
        raise ValueError(f"{name}: predicted {max(lane_h) + setup:.1f} h, above {limit} h; split it")
    lanes_txt = " and ".join(f"{h:.2f}" for h in lane_h)
    lines = [
        "# Written by scripts/make_modules.py; do not edit by hand.",
        f"# {what}.",
        f"# Predicted {sum(lane_h):.2f} GPU hours: lanes {lanes_txt} hours"
        f" ({'planning' if CALIB is None else 'measured'} speeds).",
        f"module: {name}",
        f"platform: {platform}",
        f"calibrate: {'true' if calibrate else 'false'}",
        "lanes:",
    ]
    for lane in lanes:
        for i, s in enumerate(lane):
            lead = "  - - " if i == 0 else "    - "
            lines.append(f"{lead}{{experiment: {s['experiment']}, method: {s['method']}, setting: {s['setting']}, "
                         f"seed: {s['seed']}, set: [{', '.join(flow(x) for x in s['set'])}]}}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Write the module files")
    ap.add_argument("--check", action="store_true", help="only compare with the files on disk")
    args = ap.parse_args()
    tuned_path = CONFIG_DIR / "tuned.yaml"
    tuned = load_yaml(tuned_path) if tuned_path.exists() else {}
    differ = []
    (CONFIG_DIR / "modules").mkdir(exist_ok=True)
    for name, item in plan(tuned).items():
        path = CONFIG_DIR / "modules" / f"{name}.yaml"
        text = render(name, *item)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                differ.append(name)
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(text.splitlines()[2].lstrip("# "), "->", path.name)
    if differ:
        print("Out of date:", ", ".join(differ))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
