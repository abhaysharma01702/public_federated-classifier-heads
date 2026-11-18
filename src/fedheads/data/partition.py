"""Client partitions within one language.

* ``iid_partition``: label-stratified, even split (experiments 2 and 5).
* ``dirichlet_partition``: label skew, Dirichlet(alpha) shares (experiment 3).
* ``domain_partition``: one domain per client (experiment 4, Marathi).

Every function returns ``{client_id: [uid, ...]}`` built only from training rows, is
deterministic for a given seed, and gives disjoint clients that cover all rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def client_id(language: str, k: int) -> str:
    return f"{language}_c{k}"


def iid_partition(train: pd.DataFrame, language: str, n_clients: int, seed: int) -> dict[str, list[str]]:
    """Label-stratified IID split: shuffle each label's rows, then deal them round-robin."""
    if n_clients < 1:
        raise ValueError("n_clients must be at least 1")
    rng = np.random.default_rng(seed)
    clients: dict[str, list[str]] = {client_id(language, k): [] for k in range(n_clients)}
    names = list(clients)
    offset = 0
    for label in sorted(train["label"].unique()):
        uids = np.array(sorted(train.loc[train["label"] == label, "uid"]))
        rng.shuffle(uids)
        for i, uid in enumerate(uids):
            clients[names[(i + offset) % n_clients]].append(str(uid))
        offset += len(uids)  # keeps client sizes within one row of each other
    return {k: sorted(v) for k, v in clients.items()}


def dirichlet_partition(train: pd.DataFrame, language: str, n_clients: int, alpha: float, seed: int,
                        min_size: int = 20, max_tries: int = 1000) -> dict[str, list[str]]:
    """Label skew: each label's rows are split across clients by Dirichlet(alpha) shares.
    Resamples until every client has at least ``min_size`` rows."""
    if n_clients * min_size > len(train):
        raise ValueError(f"{len(train)} rows cannot give {n_clients} clients {min_size} rows each")
    rng = np.random.default_rng(seed)
    labels = sorted(train["label"].unique())
    by_label = {lab: np.array(sorted(train.loc[train["label"] == lab, "uid"])) for lab in labels}
    for _ in range(max_tries):
        assignment: list[list[str]] = [[] for _ in range(n_clients)]
        for lab in labels:
            uids = by_label[lab].copy()
            rng.shuffle(uids)
            shares = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(uids)).astype(int)
            for k, part in enumerate(np.split(uids, cuts)):
                assignment[k].extend(str(u) for u in part)
        if min(len(a) for a in assignment) >= min_size:
            return {client_id(language, k): sorted(a) for k, a in enumerate(assignment)}
    raise RuntimeError(f"no Dirichlet draw gave every client {min_size} rows in {max_tries} tries")


def domain_partition(train: pd.DataFrame, language: str, n_clients: int, seed: int) -> dict[str, list[str]]:
    """One domain per client. With more clients than domains, each domain's rows are split
    IID among its clients; every domain gets at least one client and the extra clients go to
    the largest domains first.
    Client IDs name the domain: ``mr_movie_c0``."""
    domains = sorted(train["domain"].unique())
    if n_clients < len(domains):
        raise ValueError(f"{n_clients} clients cannot hold {len(domains)} domains one each")
    sizes = train["domain"].value_counts()
    by_size = sorted(domains, key=lambda d: (-sizes[d], d))
    extra = set(by_size[:n_clients % len(domains)])
    per = [n_clients // len(domains) + (1 if d in extra else 0) for d in domains]
    out: dict[str, list[str]] = {}
    for d, k in zip(domains, per):
        out.update(iid_partition(train[train["domain"] == d], f"{language}_{d}", k, seed))
    return out


def check_partition(partition: dict[str, list[str]], train: pd.DataFrame) -> None:
    """Raise if clients overlap or do not cover every training row exactly once."""
    seen: set[str] = set()
    for cid, uids in partition.items():
        overlap = seen.intersection(uids)
        if overlap:
            raise AssertionError(f"client {cid} shares {len(overlap)} rows with earlier clients")
        seen.update(uids)
    expected = set(train["uid"])
    if seen != expected:
        raise AssertionError(
            f"partition covers {len(seen)} rows, expected {len(expected)} "
            f"(missing {len(expected - seen)}, extra {len(seen - expected)})"
        )
