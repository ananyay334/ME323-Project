"""Small shared helpers: logging, seeding, table output."""

from __future__ import annotations

import logging
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("tribo_final")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    LOG.setLevel(level)
    LOG.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        LOG.addHandler(fh)
    return LOG


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def quiet_warnings() -> None:
    """Silence noisy numerical warnings from tiny-sample fits (convergence, all-NaN slices)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def fmt_num(x, digits: int = 3) -> str:
    """Compact number formatting for reports (handles NaN and scientific values)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "–"
    if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{max(digits - 1, 1)}e}"
    return f"{x:.{digits}g}" if abs(x) >= 1 else f"{x:.{digits}f}".rstrip("0").rstrip(".") if x != 0 else "0"
