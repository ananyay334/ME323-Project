"""Per-run file layout (config/file_layout.yaml): file roles, scan location, checkpoint,
and ``organize`` (flat dump -> data/raw/<RunID>/).

A *scan* is one height measurement: the height file plus its companions (preview image,
metadata) that share its name. Every scan gets a location (m, a, b, c) and a checkpoint
(base, cp1, cp2) by the first rule of the checkpoint ladder that assigns all scans of
that location uniquely (name token -> sub-folder -> metadata timestamp -> mtime order).
"""
from __future__ import annotations

import datetime as _dt
import filecmp
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_ORDER = ["ignore", "sem", "recipe", "cof_csv", "height", "preview", "metadata"]
CHECKPOINTS = ["base", "cp1", "cp2"]
BY_COUNT = {1: ["cp2"], 2: ["cp1", "cp2"], 3: ["base", "cp1", "cp2"]}


def file_role(name: str, layout: dict) -> str | None:
    for role in ROLE_ORDER:
        pat = layout["roles"].get(role)
        if pat and re.search(pat, name, re.I):
            return role
    return None


def run_id_of(name: str, layout: dict) -> str | None:
    m = re.match(layout["run_id_regex"], name, re.I)
    return m.group(1).upper() if m else None


def location_of(name: str, layout: dict) -> str:
    m = re.match(layout["location"]["regex"], name, re.I)
    return m.group(1).lower() if m else layout["location"].get("default", "m")


def scan_key(path: Path, layout: dict) -> str:
    """Name shared by a scan's files: no extensions, no preview/metadata suffix."""
    stem = path.name.split(".")[0]
    return re.sub(layout["checkpoint"]["companion_suffix_regex"], "", stem, flags=re.I)


def _remainder(key: str, layout: dict) -> str:
    """Part of a scan name after '<RunID>[_<loc>]' (where checkpoint tokens may sit)."""
    m = re.match(layout["location"]["regex"], key, re.I) or re.match(layout["run_id_regex"], key, re.I)
    return key[m.end():] if m else key


def checkpoint_token(text: str, layout: dict, t_run: float = np.nan) -> str | None:
    """'base' | 'cp1' | 'cp2' from a name token, or None."""
    tok = layout["checkpoint"]["tokens"]
    m = re.search(tok["seconds"], text, re.I)
    if m:
        t = float(m.group(2))
        if t == 0:
            return "base"
        if np.isfinite(t_run):
            return ["cp1", "cp2"][int(np.argmin([abs(t - t_run / 2), abs(t - t_run)]))]
    hits = [cp for cp in CHECKPOINTS if re.search(tok[cp], text, re.I)]
    return hits[0] if len(hits) == 1 else None


def metadata_time(path: Path, layout: dict) -> _dt.datetime | None:
    try:
        txt = path.read_bytes()[:8192].decode("utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(layout["checkpoint"]["metadata_time_regex"], txt)
    if not m:
        return None
    s = re.sub(r"[/.]", "-", m.group(1)[:10]) + " " + re.sub(r"[\-.]", ":", m.group(1)[11:])
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def classify_run(run_dir: Path, layout: dict, t_run: float = np.nan) -> tuple[dict, pd.DataFrame, list]:
    """Files of one run folder (recursive).

    Returns ``(files_by_role, scans, issues)``; ``scans`` has one row per scan with
    ``key, location, height, preview, metadata, checkpoint, checkpoint_source``.
    """
    run_dir = Path(run_dir)
    by_role: dict[str, list[Path]] = {r: [] for r in ROLE_ORDER}
    by_role["unknown"] = []
    for p in sorted(q for q in run_dir.rglob("*") if q.is_file()):
        by_role[file_role(p.name, layout) or "unknown"].append(p)
    scans: dict[tuple, dict] = {}
    for role in ("height", "preview", "metadata"):
        for p in by_role[role]:
            key = (p.parent, scan_key(p, layout))
            if role == "metadata" and key not in scans:
                continue                       # metadata without a scan (e.g. run log)
            s = scans.setdefault(key, {"dir": p.parent, "key": key[1], "height": None,
                                       "preview": None, "metadata": None})
            if s[role] is None:
                s[role] = p
    issues: list[str] = []
    rows = []
    for s in scans.values():
        s["location"] = location_of(s["key"], layout)
        rows.append(s)
    df = pd.DataFrame(rows, columns=["dir", "key", "height", "preview", "metadata", "location"])
    df["checkpoint"], df["checkpoint_source"] = None, None
    for loc, sub in df.groupby("location"):
        # a location with height files uses those; PNG-only scans are the fallback
        use = sub[sub["height"].notna()] if sub["height"].notna().any() else sub
        df.loc[sub.index.difference(use.index), "checkpoint_source"] = "superseded_by_height"
        cps, src = assign_checkpoints(use, run_dir, layout, t_run)
        if cps is None:
            issues.append(f"checkpoint_unresolved[{loc}: {len(use)} scans]")
            continue
        df.loc[use.index, "checkpoint"] = cps
        df.loc[use.index, "checkpoint_source"] = src
    return by_role, df, issues


def assign_checkpoints(scans: pd.DataFrame, run_dir: Path, layout: dict, t_run: float):
    """Checkpoint ladder for the scans of one location -> (list | None, rule name)."""
    n = len(scans)
    main = [r["height"] if r["height"] is not None else r["preview"] for _, r in scans.iterrows()]

    def unique(c):
        return None not in c and len(set(c)) == len(c)

    for rule in layout["checkpoint"]["ladder"]:
        if rule == "filename":
            c = [checkpoint_token(_remainder(k, layout), layout, t_run) for k in scans["key"]]
        elif rule == "subfolder":
            c = []
            for d in scans["dir"]:
                parts = Path(d).relative_to(run_dir).parts
                hits = {checkpoint_token(p, layout, t_run) for p in parts} - {None}
                c.append(hits.pop() if len(hits) == 1 else None)
        elif rule in ("metadata", "mtime"):
            if n not in BY_COUNT:
                continue
            if rule == "metadata":
                t = [metadata_time(m, layout) if m is not None else None for m in scans["metadata"]]
                if None in t or len(set(t)) < n:
                    continue
            else:
                t = [p.stat().st_mtime for p in main]
                if len(set(t)) < n:
                    continue
            order = np.argsort(np.array(t, dtype=object if rule == "metadata" else float), kind="stable")
            c = [None] * n
            for rank, i in enumerate(order):
                c[i] = BY_COUNT[n][rank]
        else:
            raise ValueError(f"unknown checkpoint rule {rule!r}")
        if unique(c):
            return c, rule
    return None, None


# ---------------------------------------------------------------------------
# organize: flat dump -> data/raw/<RunID>/
# ---------------------------------------------------------------------------
def match_load_freq(name: str, sheet: pd.DataFrame, layout: dict) -> list[str]:
    """Run IDs whose (Load_N, Freq_Hz) equal a '<load>_<freq>.csv' name."""
    m = re.match(layout["load_freq_csv_regex"], name, re.I)
    if not m:
        return []
    L, f = float(m.group(1)), float(m.group(2))
    s = sheet[pd.to_numeric(sheet["Load_N"], errors="coerce").sub(L).abs().lt(1e-6)
              & pd.to_numeric(sheet["Freq_Hz"], errors="coerce").sub(f).abs().lt(1e-6)]
    return s["Experiment_ID"].tolist()


def organize(dump_dir: Path, cfg: dict, log=print) -> tuple[list, pd.DataFrame]:
    """Copy a dump into data/raw/<RunID>/ (sub-folders kept, mtimes kept); never guess.

    Files without a run ID, with an unmatched / ambiguous ``<load>_<freq>.csv`` name, or
    whose destination already holds a different file are listed in data/raw/_unassigned.csv.
    """
    from .pipeline import load_run_sheet
    layout = cfg["layout"]
    raw = Path(cfg["paths"]["raw_dir"])
    sheet = load_run_sheet(cfg["paths"]["run_sheet"])
    if "Include" in sheet:
        sheet = sheet[sheet["Include"].astype(str).str.upper().str.strip() != "N"]
    ids = set(sheet["Experiment_ID"])
    dump_dir = Path(dump_dir)
    copied, unassigned = [], []
    for p in sorted(q for q in dump_dir.rglob("*") if q.is_file()):
        rel = p.relative_to(dump_dir)
        if file_role(p.name, layout) == "ignore":
            continue
        rid = run_id_of(p.name, layout)
        cand = [rid] if rid else []
        if rid is None and file_role(p.name, layout) == "cof_csv":
            cand = match_load_freq(p.name, sheet, layout)
        if len(cand) != 1 or cand[0] not in ids:
            why = ("ambiguous_load_freq" if len(cand) > 1 else "run_id_not_in_run_sheet" if cand
                   else "no_run_id")
            unassigned.append({"file": str(rel), "reason": why, "candidates": ";".join(cand)})
            continue
        dest = raw / cand[0] / rel.parent / p.name
        if dest.exists():
            if filecmp.cmp(p, dest, shallow=False):
                continue
            unassigned.append({"file": str(rel), "reason": "name_clash", "candidates": cand[0]})
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        copied.append((str(rel), cand[0]))
    un = pd.DataFrame(unassigned, columns=["file", "reason", "candidates"])
    raw.mkdir(parents=True, exist_ok=True)
    un.to_csv(raw / "_unassigned.csv", index=False)
    log(f"copied {len(copied)} file(s) into {raw}; {len(un)} unassigned -> {raw / '_unassigned.csv'}")
    return copied, un
