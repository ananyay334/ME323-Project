"""Configuration loading and project-path resolution."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "extraction.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load the YAML config (default: config/extraction.yaml) and apply overrides."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg = _deep_merge(cfg, overrides or {})
    # resolve relative paths against the project root
    for key, val in cfg.get("paths", {}).items():
        p = Path(val)
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p
    return cfg
