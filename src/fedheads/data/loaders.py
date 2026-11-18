"""Batches from tokenized tables.

A table has one row per sentence, with columns ``uid``, ``split`` (train, validation,
test), ``label`` (0 negative, 1 neutral, 2 positive) and ``ids`` (token ids); a
language with domains (Marathi) adds ``domain``.

* Dynamic padding: every batch is padded only to its own longest sequence.
* Client batches come only from that client's rows: no cross-client mixing.
* Batch order depends only on (run seed, client, round), so a resumed run draws
  exactly the batches an uninterrupted run would.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def drop_neutral(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """remove neutral rows (label 1) from every split; positive 2 becomes 1."""
    out = {}
    for lang, t in tables.items():
        t = t[t["label"] != 1].copy()
        t["label"] = t["label"].replace({2: 1})
        out[lang] = t.reset_index(drop=True)
    return out


def load_token_tables(tok_dir: str | Path, backbone: str, languages: list[str]) -> dict[str, pd.DataFrame]:
    return {lang: pd.read_parquet(Path(tok_dir) / backbone / f"{lang}.parquet") for lang in languages}


def pad(seqs: list[np.ndarray], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


def seeded_rng(*parts: int | str) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(str(p).encode()) for p in parts])


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device) -> "Batch":
        return Batch(self.input_ids.to(device, non_blocking=True), self.attention_mask.to(device, non_blocking=True),
                     self.labels.to(device, non_blocking=True))


@dataclass
class ClientData:
    """One client's training rows within its language's token table."""

    client_id: str
    language: str
    table: pd.DataFrame
    rows: np.ndarray                       # positions in ``table``

    @property
    def n(self) -> int:
        return len(self.rows)

    def batches(self, seed: int, round_idx: int, steps: int, batch_size: int, pad_id: int):
        """``steps`` batches for one round. Rows are drawn without repetition until
        the client's data runs out, then a new shuffle starts."""
        rng = seeded_rng("batches", seed, self.client_id, round_idx)
        need = steps * batch_size
        order = np.concatenate([rng.permutation(self.rows) for _ in range(-(-need // max(self.n, 1)))])[:need]
        ids = self.table["ids"].to_numpy()
        labels = self.table["label"].to_numpy()
        for s in range(steps):
            idx = order[s * batch_size:(s + 1) * batch_size]
            input_ids, mask = pad([ids[i] for i in idx], pad_id)
            yield Batch(input_ids, mask, torch.from_numpy(labels[idx].astype(np.int64)))


def eval_batches(table: pd.DataFrame, rows: np.ndarray, batch_size: int, pad_id: int):
    """Batches of one split, sorted by length to waste little padding."""
    seqs = table["ids"].to_numpy()[rows]
    labels = table["label"].to_numpy()[rows].astype(np.int64)
    order = np.argsort([len(s) for s in seqs], kind="stable")
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        ids, mask = pad([seqs[i] for i in idx], pad_id)
        yield Batch(ids, mask, torch.from_numpy(labels[idx]))
