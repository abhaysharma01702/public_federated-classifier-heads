"""Build clients and evaluation sets for a run from the token tables.

The setting config says how each language's training rows are split into clients:

    setting_cfg:
      partition: iid            # iid | dirichlet | domain
      alpha: 0.5                # dirichlet only
      clients: {mr: 5, en: 4, bn: 2, hi: 1}
      domain_languages: [mr]    # partition "domain": these languages split by domain, the rest IID

Each client holds rows of one language only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loaders import ClientData
from ..data.partition import check_partition, dirichlet_partition, domain_partition, iid_partition


def partition_language(train: pd.DataFrame, language: str, n_clients: int, setting: dict, seed: int) -> dict:
    kind = setting.get("partition", "iid")
    if kind == "iid":
        return iid_partition(train, language, n_clients, seed)
    if kind == "dirichlet":
        if n_clients == 1:   # one client: nothing to skew
            return iid_partition(train, language, 1, seed)
        return dirichlet_partition(train, language, n_clients, setting["alpha"], seed,
                                   min_size=setting.get("min_size", 20))
    if kind == "domain":
        if language in setting.get("domain_languages", []):
            return domain_partition(train, language, n_clients, seed)
        return iid_partition(train, language, n_clients, seed)
    raise ValueError(f"unknown partition {kind!r}")


def build_clients(tables: dict, cfg: dict) -> list[ClientData]:
    setting = cfg["setting_cfg"]
    clients = []
    for language in cfg["data"]["languages"]:
        t = tables[language]
        cols = ["uid", "label"] + (["domain"] if "domain" in t.columns else [])
        train = t.loc[t["split"].eq("train"), cols]
        n = int(setting["clients"][language])
        if n < 1:
            raise ValueError(f"{language}: every language keeps at least one client")
        part = partition_language(train, language, n, setting, cfg["seed"])
        check_partition(part, train)
        pos = pd.Series(np.arange(len(t)), index=t["uid"])
        for cid in sorted(part):
            clients.append(ClientData(cid, language, t, pos[part[cid]].to_numpy()))
    return clients


def eval_rows(tables: dict, cfg: dict) -> dict:
    """Validation and test: every row of the official split, per language."""
    out = {"validation": {}, "test": {}}
    for language in cfg["data"]["languages"]:
        t = tables[language]
        for split in out:
            out[split][language] = np.flatnonzero(t["split"].eq(split).to_numpy())
    return out
