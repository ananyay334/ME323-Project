"""Configuration loading and project paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Paths:
    root: Path
    testdata: Path
    inputs: Path
    outputs: Path

    @property
    def data(self) -> Path:
        return self.outputs / "data"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def report(self) -> Path:
        return self.outputs / "report"

    @property
    def logs(self) -> Path:
        return self.outputs / "logs"

    @property
    def cache(self) -> Path:
        return self.outputs / "cache"

    def make(self) -> None:
        for p in (self.data, self.tables, self.figures, self.report, self.logs, self.cache):
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def get_paths(cfg: dict) -> Paths:
    root = Path(cfg["_config_path"]).parent

    def resolve(p: str) -> Path:
        q = Path(p)
        return (q if q.is_absolute() else root / q).resolve()

    return Paths(
        root=root,
        testdata=resolve(cfg["paths"]["testdata"]),
        inputs=resolve(cfg["paths"]["inputs"]),
        outputs=resolve(cfg["paths"]["outputs"]),
    )


def section_hash(cfg: dict, *keys: str) -> str:
    """Short hash of some config sections, used as part of cache keys."""
    sub = {k: cfg.get(k) for k in keys}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=str).encode()).hexdigest()[:10]
