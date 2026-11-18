"""Turn the raw downloads into one clean table per language.

    python scripts/prepare_data.py --raw data_raw --out data_clean

Light CPU work (about 130k short texts); runs on the laptop. Writes
<out>/<lang>.parquet with columns uid, split, label, text, domain, and <out>/data_report.json
with the counts at every step and the MD5 of each table (checked again inside Ada jobs).

Labels are mapped by name, never by number: 0 negative, 1 neutral, 2 positive.

Cleaning, in this order, counted per language and split:
1. Text: Unicode NFC, whitespace collapsed; empty rows dropped.
2. Conflicts: a text carrying two labels within one split is dropped from that split.
3. Training duplicates: exact repeats of a text (same label) kept once.
4. Overlap: a training text that also appears in validation or test is dropped from training.
Validation and test keep their official rows apart from step 1 and 2, so they stay comparable
with the datasets' own papers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import pandas as pd

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
SPLITS = ("train", "validation", "test")

# Raw label number -> name, from each source's own documentation.
NAMES = {
    "tweeteval": {0: "negative", 1: "neutral", 2: "positive"},   # Hugging Face ClassLabel names in the parquet
    "iitp": {0: "negative", 1: "neutral", 2: "positive"},        # Hugging Face ClassLabel names in the parquet
    "sentnob": {0: "neutral", 1: "positive", 2: "negative"},     # SentNoB repo README; checked on examples
    "mahasent": {-1: "negative", 0: "neutral", 1: "positive"},   # MahaSent-MD README
}
MAHASENT_DOMAINS = {"MR": "movie", "ST": "subtitle", "GT": "tweet", "PT": "political"}   # PT as released
MAHASENT_TEXT = ("marathi_sentence", "marathi_text", "tweet")


def norm(s) -> str:
    return " ".join(unicodedata.normalize("NFC", str(s)).split())


def frame(text, label, names: dict, split: str, domain: str) -> pd.DataFrame:
    lab = pd.Series(label).map(names)
    if lab.isna().any():
        raise ValueError(f"unknown raw labels: {sorted(set(pd.Series(label)[lab.isna()]))}")
    return pd.DataFrame({"split": split, "label": lab.map(LABELS).astype("int64"),
                         "text": pd.Series(text).astype(str).values, "domain": domain})


def split_of(name: str) -> str:
    n = name.lower()
    return "train" if "train" in n else ("test" if "test" in n else "validation")


def load_raw(raw: Path) -> dict[str, pd.DataFrame]:
    out = {}
    out["en"] = pd.concat([frame(d["text"], d["label"], NAMES["tweeteval"], s, "tweet")
                           for s in SPLITS for d in [pd.read_parquet(raw / "tweeteval_sentiment" / f"{s}.parquet")]])
    out["bn"] = pd.concat([frame(d["Data"], d["Label"], NAMES["sentnob"], s, "comment")
                           for s, f in zip(SPLITS, ("Train", "Val", "Test"))
                           for d in [pd.read_csv(raw / "sentnob" / f"{f}.csv")]])
    out["hi"] = pd.concat([frame(d["text"], d["label"], NAMES["iitp"], s, dom)
                           for dom, sub in (("movie", "iitp-mr.hi"), ("product", "iitp-pr.hi"))
                           for s in SPLITS for d in [pd.read_parquet(raw / "iitp" / sub / f"{s}.parquet")]])
    parts = []
    for code, dom in MAHASENT_DOMAINS.items():
        for f in sorted((raw / "mahasent_md" / f"MahaSent_{code}").glob("*.csv")):
            d = pd.read_csv(f)
            col = next(c for c in MAHASENT_TEXT if c in d.columns)
            parts.append(frame(d[col], d["label"], NAMES["mahasent"], split_of(f.name), dom))
    out["mr"] = pd.concat(parts)
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep = {"raw": df["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict()}
    df = df.assign(text=df["text"].map(norm))
    empty = df["text"] == ""
    rep["empty"] = df[empty]["split"].value_counts().to_dict()
    df = df[~empty]
    n_lab = df.groupby(["split", "text"])["label"].transform("nunique")
    rep["conflicting_label"] = df[n_lab > 1]["split"].value_counts().to_dict()
    df = df[n_lab == 1]
    tr = df["split"] == "train"
    dup = tr & df.duplicated(["split", "text"])
    rep["train_duplicate"] = int(dup.sum())
    df = df[~dup]
    held = set(df.loc[df["split"] != "train", "text"])
    over = (df["split"] == "train") & df["text"].isin(held)
    rep["train_in_validation_or_test"] = int(over.sum())
    df = df[~over].copy()
    df["split"] = pd.Categorical(df["split"], SPLITS)
    df = df.sort_values("split", kind="stable").reset_index(drop=True)
    df["split"] = df["split"].astype(str)
    df["uid"] = [f"{s}-{i}" for s, i in zip(df["split"], df.groupby("split").cumcount())]
    rep["clean"] = df["split"].value_counts().reindex(SPLITS, fill_value=0).to_dict()
    rep["labels"] = {s: df[df["split"] == s]["label"].value_counts().sort_index().to_dict() for s in SPLITS}
    rep["domains"] = df.groupby(["domain", "split"]).size().unstack(fill_value=0).to_dict("index")
    return df[["uid", "split", "label", "text", "domain"]], rep


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=Path("data_raw"))
    ap.add_argument("--out", type=Path, default=Path("data_clean"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"label_ids": LABELS, "languages": {}}
    for lang, df in load_raw(args.raw).items():
        df, rep = clean(df)
        df.insert(0, "uid", df.pop("uid").radd(f"{lang}-"))
        path = args.out / f"{lang}.parquet"
        df.to_parquet(path, index=False)
        rep["md5"] = md5(path)
        rep["bytes"] = path.stat().st_size
        report["languages"][lang] = rep
        print(lang, rep["raw"], "->", rep["clean"])
    (args.out / "data_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=int),
                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
