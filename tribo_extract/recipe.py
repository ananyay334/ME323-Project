"""Run parameters read from the run's own files (not from the DOE sheet).

* ``<RunID>.lrx`` - RTEC recipe summary (XML). Each step: ``Fz:<N> N, t:<x> min`` and, for a
  sliding step, ``Frequency (Hz): <f>``. Load, sliding time and frequency come from the
  sliding step(s).
* ``<scan>.info`` - WLI acquisition details (XML): lens, pixel size, stage X/Y.
* fallback from the tribometer CSV: mean Fz while sliding, achieved frequency, sliding time.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def read_lrx(path: str | Path) -> list[dict]:
    """Recipe steps: ``[{'Fz_N': 32.0, 'duration_s': 84.0, 'freq_Hz': 7.5, 'sliding': True}, ...]``."""
    root = ET.parse(path).getroot()
    steps = []
    for s in root.iter("Steps"):
        txt = s.findtext("StepDetail") or ""
        d: dict = {"text": " | ".join(x.strip() for x in txt.splitlines() if x.strip())}
        m = re.search(r"Fz:\s*([-\d.]+)\s*N", txt)
        d["Fz_N"] = float(m.group(1)) if m else np.nan
        m = re.search(r"\bt:\s*([\d.]+)\s*(min|s|h)\b", txt)
        d["duration_s"] = float(m.group(1)) * {"min": 60, "s": 1, "h": 3600}[m.group(2)] if m else np.nan
        m = re.search(r"Frequency \(Hz\):\s*([\d.]+)", txt)
        d["freq_Hz"] = float(m.group(1)) if m else 0.0
        d["sliding"] = d["freq_Hz"] > 0
        steps.append(d)
    return steps


def recipe_params(path: str | Path) -> dict:
    """Load_N, Freq_Hz, Run_time_s of the sliding part of a recipe (NaN if absent)."""
    steps = [s for s in read_lrx(path) if s["sliding"]]
    if not steps:
        return {"Load_N": np.nan, "Freq_Hz": np.nan, "Run_time_s": np.nan, "n_sliding_steps": 0}
    loads = {s["Fz_N"] for s in steps}
    freqs = {s["freq_Hz"] for s in steps}
    return {"Load_N": steps[0]["Fz_N"], "Freq_Hz": steps[0]["freq_Hz"],
            "Run_time_s": round(float(np.nansum([s["duration_s"] for s in steps])), 3),
            "n_sliding_steps": len(steps),
            "recipe_consistent": len(loads) == 1 and len(freqs) == 1}


def read_info(path: str | Path) -> dict:
    """WLI acquisition details (XML ``<CRunDetail>`` block) as strings."""
    root = ET.parse(path).getroot()
    node = root.find(".//CRunDetail")
    return {c.tag: c.text for c in (node if node is not None else root)}
