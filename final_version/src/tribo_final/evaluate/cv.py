"""Validation protocols.

P1 (primary)   all runs of the target, leave-one-group-out (R04/R14/R20 = one group)
P2 (stability) the 18 training runs only, same LOGO
P3 (check)     train on the training runs, predict the 5 test runs once (they were inspected during development,
               so this is a check, not a sealed test)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut

from ..utils import quiet_warnings

PROTOCOLS = ["P1", "P2", "P3"]


def protocol_splits(df: pd.DataFrame, protocol: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) index arrays into df (row positions)."""
    idx = np.arange(len(df))
    if protocol == "P1":
        sub = idx
    elif protocol == "P2":
        sub = idx[df["role"].to_numpy() == "training"]
    elif protocol == "P3":
        tr = idx[df["role"].to_numpy() == "training"]
        te = idx[df["role"].to_numpy() == "test"]
        return [(tr, te)] if len(te) and len(tr) > 2 else []
    else:
        raise ValueError(protocol)
    groups = df["group"].to_numpy()[sub]
    if len(np.unique(groups)) < 3:
        return []
    return [(sub[tr], sub[te]) for tr, te in LeaveOneGroupOut().split(sub, groups=groups)]


def fit_predict(factory: Callable, df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> dict:
    quiet_warnings()
    m = factory()
    m.fit(df.iloc[tr], df["y"].to_numpy()[tr], df["group"].to_numpy()[tr])
    mean, sd = m.predict(df.iloc[te])
    extra = {}
    for obj in (m, getattr(m, "residual", None)):
        if obj is not None and hasattr(obj, "best"):
            extra["best"] = dict(obj.best)
    return {"pred": np.asarray(mean, float), "sd": None if sd is None else np.asarray(sd, float), "extra": extra}


def run_tasks(tasks: list[dict], n_jobs: int) -> list[dict]:
    """tasks: dicts with factory, df, tr, te (+ any labels). Heaviest first for better load balance."""
    order = sorted(range(len(tasks)), key=lambda i: -tasks[i].get("cost", 1))
    res = Parallel(n_jobs=n_jobs)(delayed(fit_predict)(tasks[i]["factory"], tasks[i]["df"], tasks[i]["tr"], tasks[i]["te"])
                                  for i in order)
    out = [None] * len(tasks)
    for i, r in zip(order, res):
        out[i] = r
    return out


def collect_oof(tasks: list[dict], results: list[dict], label_keys: list[str]) -> pd.DataFrame:
    rows = []
    for t, r in zip(tasks, results):
        df = t["df"]
        for j, i in enumerate(t["te"]):
            row = {k: t[k] for k in label_keys}
            row.update({"run_id": df["run_id"].iat[i], "role": df["role"].iat[i], "group": df["group"].iat[i],
                        "L": df["L"].iat[i], "lnf": df["lnf"].iat[i], "lnt": df["lnt"].iat[i],
                        "y": df["y"].iat[i], "pred": r["pred"][j],
                        "sd": np.nan if r["sd"] is None else r["sd"][j], "fold": t["fold"],
                        "best": str(r["extra"].get("best", ""))})
            rows.append(row)
    return pd.DataFrame(rows)
