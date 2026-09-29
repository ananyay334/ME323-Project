"""Parser for the tribometer's CSV export.

File layout observed in the sample run (Trail_Tribo.csv):

    line 1 : metadata names  <CR>  metadata values <CR><LF>   (a lone CR splits them)
    line 2 : channel header  e.g. Step, Timestamp, RecipeStep, DAQ.Fz (N), XYZ.Z Depth (mm), DAQ.COF (Fx),
    lines  : data rows (trailing comma -> empty last column)
    footer : "TotalTime" / "00:01:08.25"

Timestamp restarts from ~0 at every RecipeStep, so a continuous clock ``t_s``
is rebuilt here.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TriboRun:
    path: Path
    meta: dict = field(default_factory=dict)
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_time: str | None = None
    column_map: dict = field(default_factory=dict)   # standard name -> original header
    sample_rate_hz: float = float("nan")


def _split_meta(first_line: str) -> dict:
    parts = re.split(r"\r\n|\r|\n", first_line)
    parts = [p for p in parts if p.strip() != ""]
    if len(parts) < 2:
        return {}
    names = [n.strip() for n in parts[0].split(",")]
    values = [v.strip() for v in parts[1].split(",")]
    return {n: (values[i] if i < len(values) else "") for i, n in enumerate(names) if n}


def read_tribometer_csv(path: str | Path, column_patterns: dict) -> TriboRun:
    path = Path(path)
    raw = path.read_bytes().decode("utf-8", errors="replace")
    # Lines are CRLF terminated; the metadata block uses a bare CR internally.
    lines = raw.split("\r\n") if "\r\n" in raw else raw.split("\n")
    # data rows after the header are LF-only in the sample -> split again
    flat: list[str] = []
    for ln in lines:
        flat.extend(ln.split("\n"))

    header_idx = None
    for i, ln in enumerate(flat):
        if re.search(r"timestamp", ln, re.I) and re.search(r"cof|fz", ln, re.I):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path.name}: could not find the channel header line")

    meta = _split_meta("\r".join(flat[:header_idx])) if header_idx > 0 else {}

    body, footer = [], []
    for ln in flat[header_idx + 1:]:
        s = ln.strip().strip("\r")
        if not s:
            continue
        first = s.split(",")[0].strip()
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", first):
            body.append(s)
        else:
            footer.append(s)
    total_time = None
    for i, f in enumerate(footer):
        if f.lower().startswith("totaltime") and i + 1 < len(footer):
            total_time = footer[i + 1].strip(", ")

    df = pd.read_csv(io.StringIO("\n".join([flat[header_idx]] + body)))
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    df.columns = [str(c).strip() for c in df.columns]

    colmap: dict[str, str] = {}
    for std, pat in column_patterns.items():
        for c in df.columns:
            if c in colmap.values():
                continue
            if re.search(pat, c, re.I):
                colmap[std] = c
                break
    # "cof" pattern must not steal Fx; "fx_N" must not steal "COF (Fx)"
    if "fx_N" in colmap and "cof" in colmap.get("fx_N", "").lower():
        colmap.pop("fx_N")

    std = pd.DataFrame({k: pd.to_numeric(df[v], errors="coerce") for k, v in colmap.items()})
    for req in ("t_step_s", "cof"):
        if req not in std:
            raise ValueError(f"{path.name}: required channel '{req}' not found; headers = {list(df.columns)}")
    if "recipe_step" not in std:
        std["recipe_step"] = std.get("step", 1)
    std["recipe_step"] = pd.Series(std["recipe_step"]).ffill().bfill()
    if "z_depth_mm" in std:
        std["z_depth_um"] = std["z_depth_mm"] * 1000.0

    # continuous clock across recipe steps
    dt = np.nanmedian(np.diff(std["t_step_s"].to_numpy()))
    dt = dt if np.isfinite(dt) and dt > 0 else 0.01
    t = np.empty(len(std))
    offset, prev_step, prev_t = 0.0, None, None
    for i, (st, ts) in enumerate(zip(std["recipe_step"].to_numpy(), std["t_step_s"].to_numpy())):
        if prev_step is not None and st != prev_step:
            offset = t[i - 1] + dt - ts
        t[i] = ts + offset
        prev_step, prev_t = st, ts
    std.insert(0, "t_s", t)

    return TriboRun(path=path, meta=meta, data=std, total_time=total_time,
                    column_map=colmap, sample_rate_hz=1.0 / dt)
