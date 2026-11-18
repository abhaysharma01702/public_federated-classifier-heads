"""Tokenize the clean tables into the token tables the runs read (inside an Ada job).

    python scripts/tokenize_data.py --clean <dir with mr/en/bn/hi.parquet> --out <tok dir> \
        [--md5-from <data_report.json>]

A text longer than data.max_len keeps its first data.head_tokens tokens (with <s>) and its last
max_len - head_tokens (with </s>): the end of a long review often carries its verdict. Only long Hindi movie reviews are affected.

Writes <out>/<backbone>/<lang>.parquet (uid, split, label, domain, ids, n_tokens, truncated)
and <out>/tokens_report.json (pad_id, counts, truncation rate, library versions).
With --md5-from, every clean table's MD5 is checked against the laptop's data_report.json
first, and the script exits 1 on a mismatch. Settings come from configs/base.yaml
(model.backbones, data.max_len).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd

from fedheads.utils.io import load_base_config


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head_tail(ids: list[int], max_len: int, head: int) -> list[int]:
    """Keep the first `head` and last `max_len - head` ids of a sequence longer than max_len."""
    if len(ids) <= max_len:
        return list(ids)
    return list(ids[:head]) + list(ids[len(ids) - (max_len - head):])


def main() -> int:
    ap = argparse.ArgumentParser(description="Tokenize the clean tables")
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md5-from", type=Path, default=None)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    cfg = load_base_config()
    langs, max_len, head = cfg["data"]["languages"], cfg["data"]["max_len"], cfg["data"]["head_tokens"]

    if args.md5_from:
        want = json.loads(args.md5_from.read_text(encoding="utf-8"))["languages"]
        bad = [lang for lang in langs if md5(args.clean / f"{lang}.parquet") != want[lang]["md5"]]
        if bad:
            print(f"MD5 MISMATCH for {bad}; stopping", flush=True)
            return 1
        print("MD5 check passed for", langs, flush=True)

    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    report = {"max_len": max_len, "head_tokens": head, "truncation": "head_tail", "versions": {"transformers": transformers.__version__,
                                               "tokenizers": tokenizers.__version__}, "backbones": {}}
    for key, name in cfg["model"]["backbones"].items():
        tok = AutoTokenizer.from_pretrained(name, local_files_only=args.local_files_only)
        (args.out / key).mkdir(parents=True, exist_ok=True)
        rep = {"name": name, "pad_id": tok.pad_token_id, "languages": {}}
        for lang in langs:
            df = pd.read_parquet(args.clean / f"{lang}.parquet")
            full = tok(df["text"].tolist(), add_special_tokens=True, truncation=False)["input_ids"]
            n = np.array([len(x) for x in full])
            ids = [np.asarray(head_tail(x, max_len, head), dtype=np.int32) for x in full]
            out = df[["uid", "split", "label", "domain"]].copy()
            out["ids"] = ids
            out["n_tokens"] = n
            out["truncated"] = n > max_len
            out.to_parquet(args.out / key / f"{lang}.parquet", index=False)
            rep["languages"][lang] = {"rows": len(out), "mean_tokens": round(float(n.mean()), 1),
                                      "p95_tokens": int(np.percentile(n, 95)),
                                      "truncated_share": round(float((n > max_len).mean()), 4)}
            print(lang, rep["languages"][lang], flush=True)
        report["backbones"][key] = rep
    (args.out / "tokens_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
