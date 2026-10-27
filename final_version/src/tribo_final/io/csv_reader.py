"""RTEC friction-log CSV reader.

File layout (confirmed on the real exports):
  line 1  metadata names, a bare CR, then metadata values (Averaging, LogFrequency, TestDate, ...)
  line 2  channel header: Step, Timestamp, RecipeStep, DAQ.Fz (N), DAQ.COF (Fx), DAQ.Fx (N), XYZ.Z Depth (mm)
          [, XYZ.X Velocity (mm/s), Reciprocating.Frequency (Hz)]   (only R01-R10)
  ...     data rows; Timestamp restarts at ~0 in every RecipeStep
  end     "TotalTime" / "00:01:30.0" trailer
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = {
    "Step": "step",
    "Timestamp": "t_raw",
    "RecipeStep": "recipe_step",
    "DAQ.Fz (N)": "Fz",
    "DAQ.COF (Fx)": "cof_instr",
    "DAQ.Fx (N)": "Fx",
    "XYZ.Z Depth (mm)": "z_depth",
    "XYZ.X Velocity (mm/s)": "x_vel",
    "Reciprocating.Frequency (Hz)": "freq",
}


@dataclass
class FrictionLog:
    path: Path
    meta: dict
    data: pd.DataFrame  # standardised columns + continuous clock "t"
    fs: float  # logged sample rate (Hz)
    test_date: datetime | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_freq(self) -> bool:
        return "freq" in self.data.columns


def _split_lines(raw: bytes) -> list[bytes]:
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")


def _parse_date(text: str) -> datetime | None:
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def read_friction_csv(path: Path | str) -> FrictionLog:
    path = Path(path)
    lines = _split_lines(path.read_bytes())
    names = [s.strip() for s in lines[0].decode("utf-8", "ignore").split(",")]
    values = [s.strip() for s in lines[1].decode("utf-8", "ignore").split(",")]
    meta = dict(zip(names, values))

    header_idx = next(i for i, l in enumerate(lines) if l.lstrip().startswith(b"Step,"))
    body = [lines[header_idx]]
    for l in lines[header_idx + 1 :]:
        s = l.strip()
        if s and s[:1].isdigit() and b"," in s:  # drops blank lines and the TotalTime trailer
            body.append(l)
    df = pd.read_csv(io.BytesIO(b"\n".join(body)), skipinitialspace=True)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in CHANNELS.items() if k in df.columns})
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t_raw", "Fz", "Fx"]).reset_index(drop=True)

    try:
        averaging = float(meta.get("Averaging", 10))
        log_freq = float(meta.get("LogFrequency", 1000))
        fs = log_freq / averaging
    except ValueError:
        fs = 1.0 / np.median(np.diff(df["t_raw"].to_numpy()))
    df["t"] = continuous_clock(df["t_raw"].to_numpy(), df["recipe_step"].to_numpy(), 1.0 / fs)
    return FrictionLog(path=path, meta=meta, data=df, fs=fs, test_date=_parse_date(meta.get("TestDate", "")))


def continuous_clock(t_raw: np.ndarray, recipe_step: np.ndarray, dt: float) -> np.ndarray:
    """Timestamp restarts in each RecipeStep: offset every block by the end of the previous one."""
    t = np.asarray(t_raw, float).copy()
    offset = 0.0
    starts = np.r_[0, np.flatnonzero(np.diff(recipe_step) != 0) + 1, len(t)]
    for a, b in zip(starts[:-1], starts[1:]):
        t[a:b] = t_raw[a:b] - t_raw[a] + offset
        offset = t[b - 1] + dt
    return t
