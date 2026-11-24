"""Small helpers shared by notebooks: metric persistence, file hashing, table display."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C

METRICS = C.REPORTS / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)
TABLES = C.REPORTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return str(o)


def save_metrics(name: str, data: dict) -> None:
    path = METRICS / f"{name}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(data)
    path.write_text(json.dumps(old, indent=2, default=_default))


def load_metrics(name: str) -> dict:
    path = METRICS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    df.to_csv(TABLES / f"{name}.csv", index=index)
    return df


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
