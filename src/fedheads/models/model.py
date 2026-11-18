"""Model assembly and the trainable-state accessors.

Only the LoRA matrices and the head train; everything else is frozen. The accessors
return exactly those tensors, by name, and are what clients send to the server.
Frozen weights are never communicated or aggregated.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .heads import build_head


class SentimentModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, head: str = "linear", n_classes: int = 3,
                 dropout: float = 0.1, head_args: dict | None = None):
        super().__init__()
        self.encoder = encoder
        self.head = build_head(head, hidden_size, n_classes, dropout, **(head_args or {}))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, pooled representation)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(hidden, attention_mask)


def disable_incompatible_torchao(found_version: str | None = None) -> str | None:
    """Make PEFT ignore an installed torchao that is older than it supports.

    Kaggle's image (October 2026) ships PEFT 0.20 with torchao 0.10. While creating LoRA
    layers, PEFT asks whether torchao is available and raises ImportError for a version
    below 0.16. torchao is only for quantized models, which we never use, so PEFT is told
    it is absent. Returns a note when applied.
    """
    import sys
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import parse

    if found_version is None:
        try:
            found_version = version("torchao")
        except PackageNotFoundError:
            return None
    if parse(found_version) >= parse("0.16.0"):
        return None
    import peft.import_utils as iu

    original = getattr(iu, "is_torchao_available", None)
    if original is None:
        return None

    def unavailable() -> bool:
        return False

    for name, mod in list(sys.modules.items()):  # modules that imported the function by name
        if name.startswith("peft") and getattr(mod, "is_torchao_available", None) is original:
            setattr(mod, "is_torchao_available", unavailable)
    iu.is_torchao_available = unavailable
    return f"torchao {found_version} ignored: PEFT needs >= 0.16, and LoRA here never uses torchao"


def attach_lora(encoder: nn.Module, lora: dict) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    disable_incompatible_torchao()
    try:  # PEFT 0.20 reads torch.distributed.tensor without importing it
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        pass
    for p in encoder.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                     target_modules=list(lora["target_modules"]), bias="none")
    return get_peft_model(encoder, cfg)


def build_model(model_cfg: dict, seed: int, encoder: nn.Module | None = None,
                local_files_only: bool = False) -> SentimentModel:
    """Backbone from model_cfg (or a given encoder, for tests), LoRA, head; init seeded."""
    if encoder is None:
        from transformers import AutoModel

        name = model_cfg["backbones"][model_cfg["backbone"]]
        encoder = AutoModel.from_pretrained(name, add_pooling_layer=False, local_files_only=local_files_only)
    torch.manual_seed(seed)  # LoRA A and the head are initialized from the run seed
    encoder = attach_lora(encoder, model_cfg["lora"])
    return SentimentModel(encoder, encoder.config.hidden_size, head=model_cfg.get("head", "linear"),
                          n_classes=model_cfg.get("n_classes", 3), dropout=model_cfg.get("dropout", 0.1), head_args=model_cfg.get("head_args"))


def trainable_names(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def get_trainable_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    """Detached CPU copies of every trainable tensor, in a fixed order."""
    return OrderedDict((n, p.detach().to("cpu", copy=True).float())
                       for n, p in model.named_parameters() if p.requires_grad)


def set_trainable_state(model: nn.Module, state: dict) -> None:
    params = dict(model.named_parameters())
    names = trainable_names(model)
    if set(state) != set(names):
        raise KeyError(f"state keys differ from trainable parameters: "
                       f"missing {sorted(set(names) - set(state))[:3]}, extra {sorted(set(state) - set(names))[:3]}")
    with torch.no_grad():
        for n in names:
            params[n].copy_(state[n].to(params[n].device, params[n].dtype))


def count_parameters(model: nn.Module) -> dict:
    lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("head."))
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"lora": lora, "head": head, "frozen": frozen, "trainable": lora + head}
