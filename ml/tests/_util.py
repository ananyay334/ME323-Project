"""Test helpers (kept out of conftest.py so they can be imported unambiguously)."""
from __future__ import annotations

from pathlib import Path

from ml.config import deep_merge, load_config


def synth_config(synth_dir: Path, quick: bool = True, **overrides) -> dict:
    """Config whose 'synthetic' source points at ``synth_dir`` (serial, quick by default)."""
    base = {"data": {"synthetic": {"checkpoints": str(synth_dir / "runs_checkpoints.csv"),
                                   "cof_cycles": str(synth_dir / "cof_cycles.csv"),
                                   "topup_candidates": str(synth_dir / "topup_candidates.csv")}},
            "n_jobs": 1}
    return load_config(quick=quick, overrides=deep_merge(base, overrides))
