"""Run discovery, scan-location parsing and matching runs to the DOE table by value."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = re.compile(r"^R\d{2}$")
LOCATION = re.compile(r"^[RT]\d{2}_?([a-c])(?=$|[_. ])")
SKIP_SUFFIXES = {".raw", ".rtec", ".enc", ".png"}  # never opened


@dataclass
class ScanFile:
    stem: str
    location: str  # m, a, b or c
    bcrf: Path
    info: Path | None


@dataclass
class RunFiles:
    folder_id: str
    folder: Path
    csv: Path | None
    lrx: Path | None
    scans: list[ScanFile] = field(default_factory=list)


def scan_location(name: str) -> str:
    """'R04' -> m, 'R04a'/'R04_a' -> a, 'R01a_13' -> a, 'R01_13' -> m (trailing number = scan counter)."""
    m = LOCATION.match(Path(name).name)
    return m.group(1) if m else "m"


def _is_friction_csv(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return b"Step," in head and b"DAQ.Fx" in head


def find_runs(testdata: Path) -> list[RunFiles]:
    runs = []
    for folder in sorted(p for p in Path(testdata).iterdir() if p.is_dir() and RUN_DIR.match(p.name)):
        fid = folder.name
        csv = folder / f"{fid}.csv"
        if not csv.exists():  # fall back to any RTEC friction log, never a derived output
            cands = sorted(p for p in folder.glob("*.csv") if _is_friction_csv(p))
            csv = cands[0] if cands else None
        lrx = folder / f"{fid}.lrx"
        if not lrx.exists():
            cands = sorted(folder.glob("*.lrx"))
            lrx = cands[0] if cands else None
        scans = []
        for b in sorted(folder.glob("*.bcrf")):
            info = b.with_suffix(".info")
            scans.append(ScanFile(stem=b.stem, location=scan_location(b.name), bcrf=b, info=info if info.exists() else None))
        runs.append(RunFiles(fid, folder, csv, lrx, scans))
    return runs


def load_doe(path: Path) -> pd.DataFrame:
    doe = pd.read_csv(path)
    doe["Replicate_of"] = doe["Replicate_of"].fillna("").astype(str)
    return doe


def match_doe(params: dict, doe: pd.DataFrame, folder_id: str, tol: dict, taken: set[str] | None = None) -> str | None:
    """Return the DOE Experiment_ID whose (L, f, t) matches `params` within tolerance.

    Matching is by value; the folder name only breaks ties between identical rows (replicates).
    """
    taken = taken or set()

    def close(a, b, rel):
        return abs(a - b) <= rel * abs(b)

    ok = doe[
        doe.apply(
            lambda r: close(params["Load_N"], r.Load_N, tol["load"])
            and close(params["Freq_Hz"], r.Freq_Hz, tol["freq"])
            and close(params["Run_time_s"], r.Run_time_s, tol["time"]),
            axis=1,
        )
    ]
    if ok.empty:
        return None
    ids = list(ok["Experiment_ID"])
    if folder_id in ids:
        return folder_id
    free = [i for i in ids if i not in taken]
    return (free or ids)[0]


def replicate_groups(df: pd.DataFrame) -> pd.Series:
    """Runs with identical (L, f, t) read from the files share one group, labelled by the first run ID."""
    key = df[["Load_N", "Freq_Hz", "Run_time_s"]].round(3).astype(str).agg("|".join, axis=1)
    first = df.groupby(key)["run_id"].transform("min")
    return first.rename("group")


def group_codes(groups: pd.Series | np.ndarray) -> np.ndarray:
    return pd.factorize(pd.Series(groups).to_numpy())[0]
