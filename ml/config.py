"""Configuration, seeding, logging and run-snapshot helpers."""
from __future__ import annotations

import copy
import datetime as _dt
import importlib.util
import json
import logging
import os
import platform
import random
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent
DEFAULT_CONFIG = ML_DIR / "configs" / "default.yaml"
RESULTS_DIR = ML_DIR / "results"

log = logging.getLogger("ml")


def deep_merge(base: dict, override: dict | None) -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (lists are replaced, not merged)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, quick: bool = False,
                overrides: dict | None = None) -> dict:
    """Load the YAML config; with ``quick`` the file's ``quick:`` block is merged on top.

    The returned dict has ``cfg["quick_mode"]`` set and keeps the ``quick`` block out of
    the effective settings so that the saved snapshot shows what was actually used.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    quick_block = cfg.pop("quick", {}) or {}
    if quick:
        cfg = deep_merge(cfg, quick_block)
    cfg = deep_merge(cfg, overrides or {})
    cfg["quick_mode"] = bool(quick)
    cfg["config_path"] = str(path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def stable_seed(*parts: Any) -> int:
    """Deterministic 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch global generators.

    Library code never relies on these globals (every random draw takes an explicit
    seed); this only guards third-party code that does.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if has_module("torch"):
        import torch
        torch.manual_seed(seed)


def has_module(name: str) -> bool:
    """True if ``name`` is importable, without importing it (cheap, side-effect free)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Log to stderr and, optionally, to a file in the results folder."""
    root = logging.getLogger("ml")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False
    logging.captureWarnings(True)


def make_run_dir(base: Path | None = None, tag: str = "") -> Path:
    """Create ``ml/results/<timestamp>[_tag]/`` (with figures/ and models/)."""
    base = base or RESULTS_DIR
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / (f"{stamp}_{tag}" if tag else stamp)
    i = 1
    while out.exists():
        out = base / (f"{stamp}_{tag}_{i}" if tag else f"{stamp}_{i}")
        i += 1
    (out / "figures").mkdir(parents=True)
    (out / "models").mkdir()
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:                                            # noqa: BLE001
        return None


def package_versions() -> dict[str, str | None]:
    """Versions of core and optional dependencies (None = not installed)."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml", "joblib",
                 "torch", "xgboost", "shap", "efficient_kan", "kan", "tabpfn", "pysr"]:
        if not has_module(name):
            out[name] = None
            continue
        try:
            mod = __import__(name) if name not in ("pysr", "tabpfn") else None
            out[name] = getattr(mod, "__version__", "installed") if mod else "installed"
        except Exception:                                        # noqa: BLE001
            out[name] = "import-error"
    return out


def save_snapshot(cfg: dict, out_dir: Path, argv: list[str] | None = None) -> None:
    """Write the resolved config and an environment record next to the results."""
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    env = {"argv": argv or sys.argv, "git_commit": git_commit(), "platform": platform.platform(),
           "machine": platform.machine(), "versions": package_versions(),
           "time": _dt.datetime.now().isoformat(timespec="seconds")}
    with open(out_dir / "environment.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)


def patch_loky_tracker_handoff() -> None:
    """Silence a joblib<1.5 + Python 3.13 shutdown bug (no effect on results).

    loky hands each worker the *parent's* multiprocessing resource-tracker pid; Python
    3.13's new ``ResourceTracker.__del__`` then ``waitpid``s that pid at worker exit and
    prints a ``ChildProcessError`` traceback per worker. The worker keeps the tracker fd
    (registration still works); only the pid is withheld so the destructor skips the
    bogus wait. Applied in the parent before workers are spawned; idempotent.
    """
    if sys.version_info < (3, 13):
        return
    try:
        import joblib
        from joblib.externals.loky.backend import spawn as _spawn
    except Exception:                                            # noqa: BLE001
        return
    ver = tuple(int(x) for x in joblib.__version__.split(".")[:2] if x.isdigit())
    if ver >= (1, 5) or getattr(_spawn.get_preparation_data, "_ml_patched", False):
        return
    orig = _spawn.get_preparation_data

    def get_preparation_data(*args, **kwargs):
        d = orig(*args, **kwargs)
        if isinstance(d.get("mp_tracker_args"), dict):
            d["mp_tracker_args"]["pid"] = None
        return d

    get_preparation_data._ml_patched = True
    _spawn.get_preparation_data = get_preparation_data


def data_paths(cfg: dict, source: str) -> dict[str, Path | list]:
    """Paths and holdout IDs for ``source`` in {'real', 'synthetic'}."""
    if source not in cfg["data"]:
        raise ValueError(f"unknown data source {source!r}; expected one of 'real', 'synthetic'")
    d = cfg["data"][source]
    return {"runs": resolve_path(d["runs"]), "timeseries": resolve_path(d["timeseries"]),
            "checkpoints": resolve_path(d["checkpoints"]) if d.get("checkpoints") else None,
            "holdout_ids": [str(x) for x in (d.get("holdout_ids") or [])]}
