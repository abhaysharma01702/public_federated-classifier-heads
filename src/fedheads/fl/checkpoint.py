"""Crash-recovery checkpoints. Written atomically; the last two kept."""

from __future__ import annotations

import os
from pathlib import Path

import torch

NAME = "checkpoint_r{:04d}.pt"


def save_checkpoint(run_dir, payload: dict, keep: int = 2) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / NAME.format(payload["round"])
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    for old in sorted(run_dir.glob("checkpoint_r*.pt"))[:-keep]:
        old.unlink()
    return path


def load_checkpoint(run_dir) -> dict | None:
    found = sorted(Path(run_dir).glob("checkpoint_r*.pt"))
    if not found:
        return None
    return torch.load(found[-1], map_location="cpu", weights_only=False)


def clear_checkpoints(run_dir) -> None:
    for p in Path(run_dir).glob("checkpoint_r*.pt"):
        p.unlink()
