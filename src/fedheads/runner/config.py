"""Config layering and run identity.

base.yaml, then configs/method/<method>.yaml, then configs/setting/<setting>.yaml,
then explicit overrides. The run ID is a readable prefix plus 6 characters of a hash
of the sorted resolved config, which includes ``code_rev``: changed code gives new run
IDs instead of mixing with old results.
"""

from __future__ import annotations

import copy
import hashlib
import json

from ..utils.io import CONFIG_DIR, load_base_config, load_yaml


def deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(experiment: str, method: str, setting: str, seed: int, overrides: dict | None = None) -> dict:
    cfg = load_base_config()
    cfg = deep_update(cfg, load_yaml(CONFIG_DIR / "method" / f"{method}.yaml"))
    cfg = deep_update(cfg, load_yaml(CONFIG_DIR / "setting" / f"{setting}.yaml"))
    cfg = deep_update(cfg, overrides or {})
    cfg.update({"experiment": experiment, "method": method, "setting": setting, "seed": int(seed)})
    cfg["run_id"] = run_id(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    body = {k: v for k, v in cfg.items() if k != "run_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:6]


def run_id(cfg: dict) -> str:
    return (f"{cfg['experiment']}-{cfg['setting']}-{cfg['method']}-{cfg['model']['head']}"
            f"-s{cfg['seed']}-{config_hash(cfg)}")
