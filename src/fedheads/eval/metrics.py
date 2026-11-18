"""Macro-F1 per language (and per Marathi domain).

Model selection uses only the validation sets: the mean validation macro-F1 over
languages, each language counted equally.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score

from ..data.loaders import eval_batches


def macro_f1(y_true, y_pred, n_classes: int = 3) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


class Evaluator:
    """Holds the evaluation rows per (split, language); runs the model on them."""

    def __init__(self, tables: dict, split_rows: dict, pad_id: int, device, batch_size: int = 128,
                 use_amp: bool = False, n_classes: int = 3):
        # split_rows: {split: {language: np.ndarray of table positions}}
        self.tables = tables
        self.split_rows = split_rows
        self.pad_id = pad_id
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.n_classes = n_classes

    @torch.no_grad()
    def predict(self, model, language: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        preds, golds = [], []
        for batch in eval_batches(self.tables[language], rows, self.batch_size, self.pad_id):
            b = batch.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits, _ = model(b.input_ids, b.attention_mask)
            preds.append(logits.argmax(-1).cpu().numpy())
            golds.append(b.labels.cpu().numpy())
        return np.concatenate(golds), np.concatenate(preds)

    def evaluate(self, model, split: str) -> list[dict]:
        rows = []
        for language, idx in self.split_rows[split].items():
            y, p = self.predict(model, language, idx)
            rows.append({"split": split, "language": language, "macro_f1": macro_f1(y, p, self.n_classes),
                         "accuracy": float((y == p).mean()), "n": int(len(y))})
        return rows

    def score(self, rows: list[dict]) -> float:
        return mean_f1(rows)


def mean_f1(rows: list[dict]) -> float:
    vals = [r["macro_f1"] for r in rows]
    return float(np.mean(vals)) if vals else float("nan")
