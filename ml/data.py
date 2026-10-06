"""Load and clean the two ML input tables; target definitions and censoring.

Data rules (see ml/README.md):
* drop ``status == "error"`` rows and IDs starting with ``TRIAL``;
* ``status == "partial"`` rows are kept and used only for the targets they have;
* candidate input columns that are absent or all-NaN are dropped (and logged);
* rows missing a design variable (carbon, load, frequency) are dropped (logged);
* ``Wear_detected == False`` is censoring (< LOD), never zero: excluded from the wear
  targets by default, or imputed at LOD/2 for the sensitivity run;
* unseen-operating-point runs (config ``holdout_ids``) are split off here and never
  reach CV or tuning; a holdout ID pulls its whole replicate group with it;
* ``Zdepth_um`` is discarded on load: it is a diagnostic, never a feature or target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_paths
from .splits import assign_groups

log = logging.getLogger("ml.data")

TS_KEEP = ["Experiment_ID", "Time_s", "COF", "COF_std", "Phase"]


@dataclass(frozen=True)
class TargetSpec:
    """A modelling target: where it comes from and how it is transformed."""
    name: str
    column: str
    regime: str                        # "run" (one row per run) | "time" (one row per second)
    family: str                        # "wear" | "cof" | "trajectory"
    transform: str | None = None       # None | "log10"
    detected_column: str | None = None
    lod_column: str | None = None
    label: str = ""

    @property
    def is_log10(self) -> bool:
        return self.transform == "log10"

    @property
    def censorable(self) -> bool:
        return self.detected_column is not None and self.lod_column is not None

    def forward(self, v):
        v = np.asarray(v, float)
        if self.transform == "log10":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
        return v

    def inverse(self, y):
        y = np.asarray(y, float)
        return 10.0 ** y if self.transform == "log10" else y


TARGETS: dict[str, TargetSpec] = {t.name: t for t in [
    TargetSpec("log10_k", "Specific_Wear_Rate_mm3_per_Nm", "run", "wear", "log10",
               "Wear_detected", "Specific_Wear_Rate_LOD_mm3_per_Nm", "log10 k [mm³/(N·m)]"),
    TargetSpec("log10_V", "Wear_Volume_mm3", "run", "wear", "log10",
               "Wear_detected", "Wear_Volume_LOD_mm3", "log10 V [mm³]"),
    TargetSpec("COF_ss_mean", "COF_ss_mean", "run", "cof", None, label="steady-state COF"),
    TargetSpec("COF_t", "COF", "time", "cof", None, label="COF(t)"),
    TargetSpec("log10_Vdot_steady", "Vdot_steady_mm3_per_s", "run", "trajectory", "log10",
               label="log10 V̇_steady [mm³/s]"),
    TargetSpec("log10_dV_run", "dV_run_mm3", "run", "trajectory", "log10",
               label="log10 ΔV_run [mm³]"),
    TargetSpec("log10_tau_trans", "tau_trans_s", "run", "trajectory", "log10",
               label="log10 τ_trans [s]"),
]}
# derived run-level score of the COF(t) models (predicted COF(t) averaged over the steady phase)
COF_T_RUN = "COF_t@ss"


@dataclass
class TriboData:
    """Cleaned tables. ``runs`` holds every usable run (CV + holdout) with a ``group`` column."""
    runs: pd.DataFrame
    ts: pd.DataFrame | None
    checkpoints: pd.DataFrame | None
    inputs: list[str]
    dropped_inputs: dict[str, str]
    holdout_ids: list[str]
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def cv_runs(self) -> pd.DataFrame:
        return self.runs[~self.runs["is_holdout"]]

    @property
    def holdout_runs(self) -> pd.DataFrame:
        return self.runs[self.runs["is_holdout"]]

    def note(self, msg: str, level: int = logging.INFO) -> None:
        self.notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)


def parse_bool(s: pd.Series) -> pd.Series:
    """'True'/'False'/1/0/bool -> boolean with NA for anything else."""
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False,
         "yes": True, "no": False}
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else m.get(str(v).strip().lower(),
                                                                          pd.NA)).astype("boolean")


def _read_runs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. For real data run `python -m tribo_extract extract`; for "
            f"synthetic data run `python -m ml.synthetic` and pass `--data synthetic`.")
    return pd.read_csv(path, dtype={"Experiment_ID": str, "Replicate_of": str, "status": str,
                                    "issues": str, "cof_flags": str, "wear_flags": str})


def load_data(cfg: dict, source: str) -> TriboData:
    """Load, clean and annotate the tables of ``source`` ('real' | 'synthetic')."""
    paths = data_paths(cfg, source)
    dcfg = cfg["data"]
    notes: list[str] = []

    def note(msg, level=logging.INFO):
        notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)

    runs = _read_runs(paths["runs"])
    runs["Experiment_ID"] = runs["Experiment_ID"].astype(str).str.strip()
    n0 = len(runs)
    if runs["Experiment_ID"].duplicated().any():
        dup = runs.loc[runs["Experiment_ID"].duplicated(), "Experiment_ID"].tolist()
        note(f"duplicate Experiment_IDs {dup}: keeping the last row of each", logging.WARNING)
        runs = runs.drop_duplicates("Experiment_ID", keep="last")
    status = runs.get("status", pd.Series("ok", index=runs.index)).astype(str).str.strip().str.lower()
    bad_status = status.isin([s.lower() for s in dcfg["drop_status"]])
    bad_prefix = runs["Experiment_ID"].str.startswith(tuple(dcfg["drop_id_prefixes"]))
    if bad_status.any():
        note(f"dropped {int(bad_status.sum())} run(s) with status in {dcfg['drop_status']}: "
             f"{runs.loc[bad_status, 'Experiment_ID'].tolist()}")
    if (bad_prefix & ~bad_status).any():
        note(f"dropped {int((bad_prefix & ~bad_status).sum())} run(s) with ID prefix in "
             f"{dcfg['drop_id_prefixes']}: {runs.loc[bad_prefix & ~bad_status, 'Experiment_ID'].tolist()}")
    runs = runs[~(bad_status | bad_prefix)].copy()
    if runs.empty:
        raise ValueError(f"no usable runs in {paths['runs']}: all {n0} row(s) were dropped "
                         f"(status in {dcfg['drop_status']} or ID prefix in "
                         f"{dcfg['drop_id_prefixes']}). Run `python -m tribo_extract extract` on "
                         "the campaign data first, or use `--data synthetic`.")
    if "Replicate_of" not in runs:
        runs["Replicate_of"] = np.nan
    runs["Replicate_of"] = runs["Replicate_of"].replace({"nan": np.nan, "": np.nan})

    # ---- inputs: numeric, drop absent / all-NaN --------------------------------
    dropped: dict[str, str] = {}
    inputs: list[str] = []
    for col in dcfg["input_columns"]:
        if col not in runs:
            dropped[col] = "absent"
            continue
        runs[col] = pd.to_numeric(runs[col], errors="coerce")
        if runs[col].isna().all():
            dropped[col] = "all NaN"
        else:
            inputs.append(col)
    if dropped:
        note("dropped input columns: " + ", ".join(f"{c} ({why})" for c, why in dropped.items()))
    missing_design = [c for c in dcfg["design_columns"] if c not in inputs]
    if missing_design:
        raise ValueError(f"design column(s) {missing_design} are absent or empty in {paths['runs']}; "
                         "fill them in data/run_sheet.csv and re-run the extraction")
    no_design = runs[dcfg["design_columns"]].isna().any(axis=1)
    if no_design.any():
        note(f"dropped {int(no_design.sum())} run(s) missing a design input "
             f"({'/'.join(dcfg['design_columns'])}): {runs.loc[no_design, 'Experiment_ID'].tolist()}",
             logging.WARNING)
        runs = runs[~no_design].copy()
    if runs.empty:
        raise ValueError(f"no usable runs in {paths['runs']} ({n0} rows before cleaning). "
                         "Has the extraction been run on the campaign data?")
    for col in ["COF_ss_mean", "t_runin_s"] + [t.column for t in TARGETS.values()
                                                if t.regime == "run"] + \
               [t.lod_column for t in TARGETS.values() if t.lod_column]:
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    if "Wear_detected" in runs:
        runs["Wear_detected"] = parse_bool(runs["Wear_detected"])
    for col in inputs:
        if runs[col].isna().any():
            note(f"{col}: {int(runs[col].isna().sum())} missing value(s) (median-imputed inside each "
                 f"training fold when used as a feature)")

    # ---- groups and holdout -----------------------------------------------------
    runs["group"] = assign_groups(runs)
    hold = set(paths["holdout_ids"])
    unknown = sorted(hold - set(runs["Experiment_ID"]))
    if unknown:
        note(f"holdout IDs not found among usable runs: {unknown}", logging.WARNING)
    hold_groups = set(runs.loc[runs["Experiment_ID"].isin(hold), "group"])
    runs["is_holdout"] = runs["group"].isin(hold_groups)
    pulled = sorted(set(runs.loc[runs["is_holdout"], "Experiment_ID"]) - hold)
    if pulled:
        note(f"runs {pulled} share a replicate group with a holdout run -> also held out",
             logging.WARNING)
    holdout_ids = sorted(runs.loc[runs["is_holdout"], "Experiment_ID"])
    _warn_design_issues(runs, dcfg, note)
    runs = runs.sort_values("Experiment_ID").reset_index(drop=True)
    note(f"{len(runs)} usable runs ({len(runs) - len(holdout_ids)} for CV in "
         f"{runs.loc[~runs['is_holdout'], 'group'].nunique()} groups, "
         f"{len(holdout_ids)} unseen holdout); {n0 - len(runs)} row(s) dropped")

    # ---- COF(t) table -----------------------------------------------------------
    ts = None
    if paths["timeseries"].exists():
        raw = pd.read_csv(paths["timeseries"], dtype={"Experiment_ID": str, "Phase": str})
        raw["Experiment_ID"] = raw["Experiment_ID"].astype(str).str.strip()
        keep = [c for c in TS_KEEP if c in raw]                 # drops Zdepth_um and repeated inputs
        ts = raw[keep].copy()
        for c in ("Time_s", "COF", "COF_std"):
            if c in ts:
                ts[c] = pd.to_numeric(ts[c], errors="coerce")
        n_ts = ts["Experiment_ID"].nunique()
        ts = ts[ts["Experiment_ID"].isin(runs["Experiment_ID"])]
        ts = ts.dropna(subset=["Time_s", "COF"])
        if "Phase" not in ts:
            ts["Phase"] = np.nan
        ts = ts.merge(runs[["Experiment_ID", "group", "is_holdout"]], on="Experiment_ID")
        ts = ts.sort_values(["Experiment_ID", "Time_s"]).reset_index(drop=True)
        note(f"COF(t): {len(ts)} rows from {ts['Experiment_ID'].nunique()} runs "
             f"({n_ts - ts['Experiment_ID'].nunique()} run(s) in the file not usable)")
        if ts.empty:
            ts = None
    else:
        note(f"{paths['timeseries']} not found: COF(t) targets unavailable", logging.WARNING)

    checkpoints = None
    if paths["checkpoints"] is not None and paths["checkpoints"].exists():
        cp = pd.read_csv(paths["checkpoints"], dtype={"Experiment_ID": str})
        if {"Experiment_ID", "t_s", "Wear_Volume_mm3"} <= set(cp.columns) and len(cp):
            checkpoints = cp[cp["Experiment_ID"].isin(runs["Experiment_ID"])].copy()

    return TriboData(runs=runs, ts=ts, checkpoints=checkpoints, inputs=inputs,
                     dropped_inputs=dropped, holdout_ids=holdout_ids, source=source, notes=notes)


def _warn_design_issues(runs: pd.DataFrame, dcfg: dict, note) -> None:
    """Warn about off-grid runs used in CV and unmarked repeats of a grid condition."""
    grid = dcfg.get("grid") or {}
    cv = runs[~runs["is_holdout"]]
    if grid:
        on = np.ones(len(cv), bool)
        for col, levels in grid.items():
            if col in cv:
                on &= np.isclose(cv[col].to_numpy(float)[:, None], np.asarray(levels, float)).any(1)
        if (~on).any():
            note(f"off-grid run(s) {cv.loc[~on, 'Experiment_ID'].tolist()} are not in holdout_ids "
                 "and will be used in CV - add them to data.<source>.holdout_ids if they are the "
                 "unseen operating points", logging.WARNING)
    key = [c for c in dcfg["design_columns"]]
    per_cond = cv.groupby(key)["group"].nunique()
    if (per_cond > 1).any():
        note(f"{int((per_cond > 1).sum())} condition(s) were run more than once without "
             "`Replicate_of`; they are treated as independent groups", logging.WARNING)


# ---------------------------------------------------------------------------
def target_frame(data: TriboData, target: str | TargetSpec, censor_mode: str = "exclude",
                 split: str = "cv") -> tuple[pd.DataFrame, dict]:
    """Rows usable for ``target`` with columns ``y``, ``censored``, ``y_lod`` (+ ids, inputs).

    Run regime: one row per run. Time regime: one row per second, with the run's inputs
    joined from the runs table (single source of truth for inputs).
    """
    spec = TARGETS[target] if isinstance(target, str) else target
    runs = data.runs[data.runs["is_holdout"] == (split == "holdout")]
    info: dict = {"target": spec.name, "split": split, "censor_mode": censor_mode}
    if spec.regime == "time":
        if data.ts is None:
            return pd.DataFrame(), {**info, "n_rows": 0, "n_runs": 0}
        ts = data.ts[data.ts["Experiment_ID"].isin(runs["Experiment_ID"])]
        df = ts.drop(columns=["group", "is_holdout"]).merge(runs, on="Experiment_ID", how="inner")
        df["y"] = spec.forward(df[spec.column])
        df["censored"] = False
        df["y_lod"] = np.nan
        df = df[np.isfinite(df["y"])].reset_index(drop=True)
        info.update(n_rows=len(df), n_runs=df["Experiment_ID"].nunique())
        return df, info

    if spec.column not in runs:
        return pd.DataFrame(), {**info, "n_rows": 0, "n_runs": 0, "absent": True}
    df = runs.copy()
    val = pd.to_numeric(df[spec.column], errors="coerce").to_numpy(float)
    y = spec.forward(val)
    valid = np.isfinite(y)
    n_nonpos = int((np.isfinite(val) & ~valid).sum())
    censored = np.zeros(len(df), bool)
    y_lod = np.full(len(df), np.nan)
    if spec.censorable and spec.detected_column in df:
        det = df[spec.detected_column]
        not_det = (det == False).fillna(False).to_numpy(bool)          # noqa: E712
        lod = pd.to_numeric(df.get(spec.lod_column), errors="coerce").to_numpy(float) \
            if spec.lod_column in df else np.full(len(df), np.nan)
        y_lod = spec.forward(lod)
        censored = not_det & ~valid
        info["n_censored"] = int(censored.sum())
        info["n_censored_no_lod"] = int((censored & ~np.isfinite(y_lod)).sum())
        if censor_mode == "lod_half":
            imp = censored & np.isfinite(y_lod)
            y = np.where(imp, spec.forward(lod / 2.0), y)
            valid = valid | imp
        elif censor_mode != "exclude":
            raise ValueError(f"censoring mode must be 'exclude' or 'lod_half', got {censor_mode!r}")
    df["y"], df["censored"], df["y_lod"] = y, censored, y_lod
    info.update(n_nonpositive=n_nonpos, n_missing=int((~valid & ~censored).sum()))
    df = df[valid].reset_index(drop=True)
    info.update(n_rows=len(df), n_runs=len(df))
    return df, info
