"""RTEC recipe (.lrx, XML) parser.

Each <StepDetail> reads like
  1> Fz:32 N,  t:0.03333 min  None                                  (static step)
  2> Fz:32 N,  t:1.4 min  R - Continuous  Frequency (Hz): 7.5  ...  (sliding step)
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

_FZ = re.compile(r"Fz:\s*([-\d.]+)\s*N")
_T = re.compile(r"t:\s*([-\d.]+)\s*(min|s)")
_F = re.compile(r"Frequency\s*\(Hz\):\s*([-\d.]+)")
_ACC = re.compile(r"Acceleration\s*\(s\):\s*([-\d.]+)")
_DEC = re.compile(r"Deceleration\s*\(s\):\s*([-\d.]+)")


def parse_step(text: str) -> dict:
    fz = _FZ.search(text)
    t = _T.search(text)
    f = _F.search(text)
    t_s = None
    if t:
        t_s = float(t.group(1)) * (60.0 if t.group(2) == "min" else 1.0)
    return {
        "Fz_N": float(fz.group(1)) if fz else None,
        "t_s": t_s,
        "freq_Hz": float(f.group(1)) if f else None,
        "accel_s": float(_ACC.search(text).group(1)) if _ACC.search(text) else None,
        "decel_s": float(_DEC.search(text).group(1)) if _DEC.search(text) else None,
        "sliding": bool(f) and float(f.group(1)) > 0,
        "text": " ".join(text.split()),
    }


def parse_lrx(path: Path | str) -> dict:
    root = ET.parse(path).getroot()
    steps = [parse_step(el.text or "") for el in root.iter("StepDetail")]
    sliding = [s for s in steps if s["sliding"]]
    if not sliding:
        raise ValueError(f"{path}: no sliding step in recipe")
    return {
        "steps": steps,
        "Load_N": sliding[0]["Fz_N"],
        "Freq_Hz": sliding[0]["freq_Hz"],
        "Run_time_s": float(round(sum(s["t_s"] for s in sliding))),
        "static_s": float(sum(s["t_s"] or 0 for s in steps if not s["sliding"])),
        "source": "lrx",
    }
