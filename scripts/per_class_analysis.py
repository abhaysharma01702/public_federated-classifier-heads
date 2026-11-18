"""Per-class F1 and confusion matrices on test, from a saved best_state.pt (laptop GPU).

Usage: python scripts/per_class_analysis.py <run_dir> [<run_dir> ...] [--out file.json]

Also reports the two-class rescoring of the test rows whose gold label is
negative or positive, the prediction is whichever of those two logits is larger (neutral
ignored); macro-F1 over the two classes. Compared with the three-class model's own
predictions on the same rows.

Rebuilds the model from the run's config.resolved.yaml, loads best_state.pt, predicts
every test row of each language, and checks the macro-F1 against the run's final_test.csv
(state 'best') before reporting, so a mismatch shows at once. Needs xlm-roberta-base in
the local Hugging Face cache (no download).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedheads.models.model import build_model, set_trainable_state  # noqa: E402

LANGS = ["mr", "en", "bn", "hi"]
NAMES = ["negative", "neutral", "positive"]


@torch.no_grad()
def predict(model, tok, texts, max_len, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def analyse(run_dir: Path, device) -> dict:
    from transformers import AutoTokenizer

    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))
    mcfg = cfg["model"]
    name = mcfg["backbones"][mcfg["backbone"]]
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = build_model(mcfg, seed=0, local_files_only=True)
    set_trainable_state(model, torch.load(run_dir / "best_state.pt", map_location="cpu"))
    model.to(device)
    saved = pd.read_csv(run_dir / "final_test.csv")
    saved = saved[saved.state == "best"].set_index("language")["macro_f1"]
    max_len = cfg.get("data", {}).get("max_len", 128)
    res = {"run": run_dir.name, "languages": {}}
    for lang in LANGS:
        d = pd.read_parquet(ROOT / "data_clean" / f"{lang}.parquet")
        d = d[d.split == "test"]
        y = d.label.to_numpy()
        logits = predict(model, tok, d.text.tolist(), max_len, device)
        p = logits.argmax(-1)
        keep = y != 1
        yb, lb = y[keep], logits[keep]
        pb = np.where(lb[:, 0] > lb[:, 2], 0, 2)
        binary = {"n": int(keep.sum()),
                  "macro_f1_two_class": float(f1_score(yb, pb, labels=[0, 2], average="macro", zero_division=0)),
                  "accuracy_two_class": float((yb == pb).mean()),
                  "three_class_pred_on_same_rows_accuracy": float((yb == p[keep]).mean()),
                  "share_predicted_neutral_on_these_rows": float((p[keep] == 1).mean())}
        f1 = float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
        pr, rc, f, n = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        res["languages"][lang] = {
            "macro_f1": f1, "saved_macro_f1": float(saved.get(lang, float("nan"))),
            "per_class": {NAMES[k]: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f[k]),
                                     "n": int(n[k])} for k in range(3)},
            "confusion": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
            "pred_share": [float((p == k).mean()) for k in range(3)],
            "binary": binary,
        }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allres = []
    for r in a.runs:
        res = analyse(Path(r), device)
        allres.append(res)
        print(f"\n== {res['run']}")
        for lang, v in res["languages"].items():
            pc = v["per_class"]
            print(f"{lang}: macro {v['macro_f1']:.4f} (saved {v['saved_macro_f1']:.4f}) | "
                  + " | ".join(f"{k[:3]} F1 {pc[k]['f1']:.3f} P {pc[k]['precision']:.3f} R {pc[k]['recall']:.3f} n {pc[k]['n']}"
                               for k in NAMES))
            print(f"    confusion (rows gold neg/neu/pos): {v['confusion']}")
            b = v["binary"]
            print(f"    two-class: macro {b['macro_f1_two_class']:.4f} acc {b['accuracy_two_class']:.4f} n {b['n']} "
                  f"(three-class acc on same rows {b['three_class_pred_on_same_rows_accuracy']:.4f}, "
                  f"called neutral {b['share_predicted_neutral_on_these_rows']:.3f})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(allres, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
