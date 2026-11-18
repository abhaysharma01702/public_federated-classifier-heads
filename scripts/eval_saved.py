"""Re-evaluate saved models for the final analysis (17 November 2026). No training.

    python scripts/eval_saved.py --tok-dir <token tables> --runs <file listing run dirs> --out <results.json>
        [--max-rows N] [--local-files-only]

For each run directory (with config.resolved.yaml):
* best_state.pt / final_state.pt (FedAvg, centralized, ...): test macro-F1 and per-class F1 per language,
  and per Marathi domain;
* client_states/<client>_{best,final}_state.pt (local-only): each client's own model on its language's
  test set, macro-F1 and per-class F1, plus per Marathi domain for Marathi clients.
Results are appended per run, so an interrupted job can be resumed (finished runs are skipped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def per_class_f1(y, p, n_classes):
    from sklearn.metrics import f1_score
    return [float(x) for x in f1_score(y, p, labels=list(range(n_classes)), average=None, zero_division=0)]


def macro(y, p, n_classes):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(n_classes)), average="macro", zero_division=0))


def score_block(y, p, n_classes, domains=None):
    out = {"macro_f1": macro(y, p, n_classes), "per_class_f1": per_class_f1(y, p, n_classes), "n": int(len(y))}
    if domains is not None:
        out["by_domain"] = {}
        for d in sorted(set(domains)):
            m = domains == d
            out["by_domain"][str(d)] = {"macro_f1": macro(y[m], p[m], n_classes), "n": int(m.sum())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-dir", required=True, type=Path)
    ap.add_argument("--runs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=0, help="test only: evaluate at most this many rows per language")
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()

    import torch
    import yaml

    from fedheads.data.loaders import load_token_tables
    from fedheads.eval.metrics import Evaluator
    from fedheads.models.model import build_model, set_trainable_state
    from fedheads.runner.setup import eval_rows
    from fedheads.utils.io import read_json

    done = {}
    if a.out.exists():
        done = json.loads(a.out.read_text())
    run_dirs = [Path(l.strip()) for l in a.runs.read_text().splitlines() if l.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tables = None
    for rd in run_dirs:
        cfg = yaml.safe_load((rd / "config.resolved.yaml").read_text())
        rid = cfg["run_id"]
        if rid in done:
            print("skip", rid, flush=True)
            continue
        backbone = cfg["model"]["backbone"]
        if tables is None:
            tables = load_token_tables(a.tok_dir, backbone, cfg["data"]["languages"])
            pad_id = read_json(a.tok_dir / "tokens_report.json")["backbones"][backbone]["pad_id"]
            rows_all = eval_rows(tables, cfg)["test"]
            if a.max_rows:
                rows_all = {k: v[:a.max_rows] for k, v in rows_all.items()}
            ev = Evaluator(tables, {"test": rows_all}, pad_id, device, use_amp=use_amp)
        n_classes = cfg["model"].get("n_classes", 3)
        model = build_model(cfg["model"], cfg["seed"], local_files_only=a.local_files_only).to(device)

        def evaluate_state(state, languages):
            set_trainable_state(model, {k: v.to(device) for k, v in state.items()})
            res = {}
            for lang in languages:
                idx = rows_all[lang]
                y, p = ev.predict(model, lang, idx)
                dom = tables[lang]["domain"].to_numpy()[idx] if lang == "mr" else None
                res[lang] = score_block(y, p, n_classes, dom)
            return res

        rec = {"run_id": rid, "dir": str(rd)}
        if (rd / "best_state.pt").exists():
            for name in ("best", "final"):
                f = rd / f"{name}_state.pt"
                if f.exists():
                    rec[name] = evaluate_state(torch.load(f, map_location="cpu"), list(rows_all))
        elif (rd / "client_states").exists():
            rec["clients"] = {}
            for f in sorted((rd / "client_states").glob("*_best_state.pt")):
                cid = f.name[: -len("_best_state.pt")]
                lang = cid.split("_")[0]
                rec["clients"][cid] = {name: evaluate_state(torch.load(rd / "client_states" / f"{cid}_{name}_state.pt",
                                                                       map_location="cpu"), [lang])[lang]
                                       for name in ("best", "final")}
        else:
            rec["skipped"] = "no saved model"
        done[rid] = rec
        a.out.write_text(json.dumps(done))
        print("done", rid, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"evaluated {len(done)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
