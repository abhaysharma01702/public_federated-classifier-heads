"""Local training.

Each sampled client loads the global trainable state, builds a fresh AdamW, runs K steps
on its own data with cross-entropy, and returns its trainable state. Dropout and batch
order are seeded by (run seed, client, round), so a resumed run repeats exactly what an
uninterrupted run does on the CPU.
"""

from __future__ import annotations

import time
import zlib

import torch
import torch.nn.functional as F

from ..models.model import get_trainable_state, set_trainable_state


def make_optimizer(model, train_cfg: dict) -> torch.optim.Optimizer:
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    groups = [{"params": lora, "lr": train_cfg["lr_lora"]}, {"params": head, "lr": train_cfg["lr_head"]}]
    return torch.optim.AdamW(groups, weight_decay=train_cfg["weight_decay"])


def make_scaler(device, use_amp: bool):
    return torch.amp.GradScaler(device.type, enabled=use_amp and device.type == "cuda")


def train_steps(model, batches, optimizer, scaler, device, train_cfg: dict, use_amp: bool, before_clip=None) -> dict:
    """Run the given batches. Returns loss sums and counts for logging. The loss is
    computed in float32. ``before_clip(named_params)`` runs after unscaling, before clipping
    (FedProx adds its proximal gradient there)."""
    model.train()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    stats = {"steps": 0, "sequences": 0, "loss": 0.0}
    for batch in batches:
        b = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(b.input_ids, b.attention_mask)
        loss = F.cross_entropy(logits.float(), b.labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if before_clip is not None:
            before_clip(named)
        torch.nn.utils.clip_grad_norm_(params, train_cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        stats["steps"] += 1
        stats["sequences"] += int(b.input_ids.shape[0])
        stats["loss"] += float(loss.detach())
    return stats


def client_torch_seed(seed: int, client_id: str, round_idx: int) -> int:
    return zlib.crc32(f"dropout|{seed}|{client_id}|{round_idx}".encode())


def local_update(model, global_state: dict, client, round_idx: int, cfg: dict, device, scaler, pad_id: int,
                 use_amp: bool, before_clip=None) -> tuple[dict, dict]:
    """One client's round: returns (new trainable state, stats)."""
    t0 = time.time()
    set_trainable_state(model, global_state)
    torch.manual_seed(client_torch_seed(cfg["seed"], client.client_id, round_idx))
    optimizer = make_optimizer(model, cfg["train"])
    batches = client.batches(cfg["seed"], round_idx, cfg["train"]["local_steps"], cfg["train"]["batch_sentences"],
                             pad_id=pad_id)
    stats = train_steps(model, batches, optimizer, scaler, device, cfg["train"], use_amp, before_clip)
    stats["seconds"] = time.time() - t0
    return get_trainable_state(model), stats
