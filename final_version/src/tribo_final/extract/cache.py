"""Per-run extraction cache: one pickle per run, keyed by the input files and the relevant config."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

CODE_VERSION = "extract-v3"
_PKG = Path(__file__).resolve().parents[1]


def code_hash() -> str:
    """Hash of the reader and extraction source, so editing them invalidates the cache."""
    h = hashlib.sha1()
    for f in sorted(list((_PKG / "io").glob("*.py")) + list((_PKG / "extract").glob("*.py"))):
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def files_key(paths: list[Path | None]) -> list:
    out = []
    for p in paths:
        if p is None:
            continue
        st = Path(p).stat()
        out.append([Path(p).name, st.st_size, int(st.st_mtime)])
    return out


def cache_key(paths: list[Path | None], cfg_parts: dict) -> str:
    blob = json.dumps({"v": CODE_VERSION, "code": code_hash(), "files": files_key(paths), "cfg": cfg_parts},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load(cache_dir: Path, name: str, key: str):
    f = cache_dir / f"{name}_{key}.pkl"
    if f.exists():
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # written by another Python / pandas version: recompute
            return None
    return None


def store(cache_dir: Path, name: str, key: str, obj) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{name}_*.pkl"):
        old.unlink()
    with open(cache_dir / f"{name}_{key}.pkl", "wb") as fh:
        pickle.dump(obj, fh)
