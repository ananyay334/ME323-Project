"""Load and clean the ML input tables; the four data regimes; censoring.

| regime              | source                 | rows                | targets                          |
|---------------------|------------------------|---------------------|----------------------------------|
| `endpoint`          | runs_targets.csv       | 1 per run (40)      | log10 k, log10 V_w(t_end), COF_ss |
| `trajectory_params` | runs_targets.csv       | runs with eq.-1 fit | log10 V̇_steady, ΔV_run, τ_trans  |
| `wear_checkpoints`  | wear_trajectory.csv    | 1 per scan, t > 0   | log10 V_w(t)  (+ input log10 t)  |
| `cof_time`          | cof_timeseries.csv     | 1 per 120-s bin     | mean COF      (+ input t, bin end) |

Run-level columns (inputs, group, specimen) are joined from runs_targets.csv on
``Experiment_ID``. Rules: drop ``status == "error"`` and IDs starting with ``TRIAL``;
rows lacking the target (e.g. partial runs) are dropped for that target only; absent
or all-NaN input columns are dropped and logged; ``Wear_detected == False`` is censoring
(< LOD), never zero: excluded by default, or imputed at LOD/2 (``censoring.mode``).
``Zdepth_um`` is never read.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import data_paths
from .splits import assign_groups

log = logging.getLogger("ml.data")

REGIMES = ("endpoint", "trajectory_params", "wear_checkpoints", "cof_time")
ROW_REGIMES = ("wear_checkpoints", "cof_time")          # several rows per run


@dataclass(frozen=True)
class TargetSpec:
    """A modelling target: where it comes from and how it is transformed."""
    name: str
    column: str
    regime: str
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
    TargetSpec("log10_k", "Specific_Wear_Rate_mm3_per_Nm", "endpoint", "log10",
               "Wear_detected", "Specific_Wear_Rate_LOD_mm3_per_Nm", "log10 k [mm³/(N·m)]"),
    TargetSpec("log10_V", "Wear_Volume_mm3", "endpoint", "log10",
               "Wear_detected", "Wear_Volume_LOD_mm3", "log10 V_w(t_end) [mm³]"),
    TargetSpec("COF_ss_mean", "COF_ss_mean", "endpoint", label="steady-state COF"),
    TargetSpec("log10_Vdot_steady", "Vdot_steady_mm3_per_s", "trajectory_params", "log10",
               label="log10 V̇_steady [mm³/s]"),
    TargetSpec("log10_dV_run", "dV_run_mm3", "trajectory_params", "log10",
               label="log10 ΔV_run [mm³]"),
    TargetSpec("log10_tau_trans", "tau_trans_s", "trajectory_params", "log10",
               label="log10 τ_trans [s]"),
    TargetSpec("log10_V_t", "Wear_Volume_mm3", "wear_checkpoints", "log10",
               "Wear_detected", "Wear_Volume_LOD_mm3", "log10 V_w(t) [mm³]"),
    TargetSpec("COF_t", "COF", "cof_time", label="COF, 120-s bin mean"),
]}


@dataclass
class TriboData:
    """Cleaned tables; ``runs`` has ``group`` and ``Specimen_ID`` for every usable run."""
    runs: pd.DataFrame
    checkpoints: pd.DataFrame | None
    cof_bins: pd.DataFrame | None
    inputs: list[str]
    dropped_inputs: dict[str, str]
    source: str
    notes: list[str] = field(default_factory=list)

    def note(self, msg: str, level: int = logging.INFO) -> None:
        self.notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)


def parse_bool(s: pd.Series) -> pd.Series:
    """'True'/'False'/1/0/bool -> boolean with NA for anything else."""
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False,
         "yes": True, "no": False}
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else m.get(str(v).strip().lower(),
                                                                          pd.NA)).astype("boolean")


def load_data(cfg: dict, source: str) -> TriboData:
    """Load, clean and annotate the tables of ``source`` ('real' | 'synthetic')."""
    paths = data_paths(cfg, source)
    dcfg = cfg["data"]
    notes: list[str] = []

    def note(msg, level=logging.INFO):
        notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)

    if not paths["runs"].exists():
        raise FileNotFoundError(
            f"{paths['runs']} not found. For real data run `python -m tribo_extract extract`; "
            "for synthetic data run `python -m ml.synthetic` and pass `--data synthetic`.")
    runs = pd.read_csv(paths["runs"], dtype={"Experiment_ID": str, "Replicate_of": str,
                                             "Specimen_ID": str, "status": str})
    runs["Experiment_ID"] = runs["Experiment_ID"].astype(str).str.strip()
    n0 = len(runs)
    if runs["Experiment_ID"].duplicated().any():
        note("duplicate Experiment_IDs: keeping the last row of each", logging.WARNING)
        runs = runs.drop_duplicates("Experiment_ID", keep="last")
    status = runs.get("status", pd.Series("ok", index=runs.index)).astype(str).str.strip().str.lower()
    bad = status.isin([s.lower() for s in dcfg["drop_status"]]) | \
        runs["Experiment_ID"].str.startswith(tuple(dcfg["drop_id_prefixes"]))
    if bad.any():
        note(f"dropped {int(bad.sum())} run(s) (status in {dcfg['drop_status']} or ID prefix in "
             f"{dcfg['drop_id_prefixes']}): {runs.loc[bad, 'Experiment_ID'].tolist()}")
    runs = runs[~bad].copy()
    if runs.empty:
        raise ValueError(f"no usable runs in {paths['runs']}: all {n0} row(s) were dropped. Run "
                         "`python -m tribo_extract extract` on the campaign data first, or use "
                         "`--data synthetic`.")
    for c in ("Replicate_of", "Specimen_ID"):
        if c not in runs:
            runs[c] = np.nan
        runs[c] = runs[c].replace({"nan": np.nan, "": np.nan})

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
        note("dropped input columns: " + ", ".join(f"{c} ({w})" for c, w in dropped.items()))
    missing = [c for c in dcfg["design_columns"] if c not in inputs]
    if missing:
        raise ValueError(f"design column(s) {missing} are absent or empty in {paths['runs']}; "
                         "fill them in data/run_sheet.csv and re-run the extraction")
    no_design = runs[dcfg["design_columns"]].isna().any(axis=1)
    if no_design.any():
        note(f"dropped {int(no_design.sum())} run(s) missing a design input: "
             f"{runs.loc[no_design, 'Experiment_ID'].tolist()}", logging.WARNING)
        runs = runs[~no_design].copy()
    if runs["Specimen_ID"].isna().any():
        fill = "C" + runs["Carbon_pct"].map(lambda c: f"{c:g}")
        note(f"Specimen_ID missing for {int(runs['Specimen_ID'].isna().sum())} run(s): "
             "one specimen per carbon level assumed", logging.WARNING)
        runs["Specimen_ID"] = runs["Specimen_ID"].fillna(fill)
    for col in [t.column for t in TARGETS.values() if t.regime in ("endpoint", "trajectory_params")] \
            + [t.lod_column for t in TARGETS.values() if t.lod_column]:
        if col in runs:
            runs[col] = pd.to_numeric(runs[col], errors="coerce")
    if "Wear_detected" in runs:
        runs["Wear_detected"] = parse_bool(runs["Wear_detected"])
    runs["group"] = assign_groups(runs)
    runs = runs.sort_values("Experiment_ID").reset_index(drop=True)
    note(f"{len(runs)} usable runs in {runs['group'].nunique()} groups on "
         f"{runs['Specimen_ID'].nunique()} specimens; {n0 - len(runs)} row(s) dropped")
    ids = set(runs["Experiment_ID"])

    checkpoints = None
    if paths["wear_trajectory"].exists():
        cp = pd.read_csv(paths["wear_trajectory"], dtype={"Experiment_ID": str})
        cp["Experiment_ID"] = cp["Experiment_ID"].astype(str).str.strip()
        for c in ("Time_s", "Wear_Volume_mm3", "Wear_Volume_LOD_mm3"):
            cp[c] = pd.to_numeric(cp.get(c), errors="coerce")
        cp["Wear_detected"] = parse_bool(cp["Wear_detected"]) if "Wear_detected" in cp \
            else pd.Series(pd.NA, index=cp.index, dtype="boolean")
        checkpoints = cp[cp["Experiment_ID"].isin(ids) & (cp["Time_s"] > 0)].reset_index(drop=True)
        note(f"wear_checkpoints: {len(checkpoints)} scans with t > 0 from "
             f"{checkpoints['Experiment_ID'].nunique()} runs")
    else:
        note(f"{paths['wear_trajectory']} not found: wear_checkpoints regime unavailable",
             logging.WARNING)

    cof_bins = None
    if paths["timeseries"].exists():
        ts = pd.read_csv(paths["timeseries"], dtype={"Experiment_ID": str},
                         usecols=lambda c: c in ("Experiment_ID", "Time_s", "COF"))
        ts["Experiment_ID"] = ts["Experiment_ID"].astype(str).str.strip()
        ts = ts[ts["Experiment_ID"].isin(ids)].dropna(subset=["Time_s", "COF"])
        b = float(dcfg.get("cof_bin_s", 120))
        ts["bin_end"] = np.ceil(ts["Time_s"].astype(float) / b) * b
        cof_bins = (ts.groupby(["Experiment_ID", "bin_end"])
                    .agg(COF=("COF", "mean"), n=("COF", "size")).reset_index()
                    .rename(columns={"bin_end": "Time_s"}))
        short = cof_bins["n"] < 0.5 * b                      # bins with < half their seconds
        cof_bins = cof_bins[~short & (cof_bins["Time_s"] > 0)].reset_index(drop=True)
        note(f"cof_time: {len(cof_bins)} bins of {b:g} s from {cof_bins['Experiment_ID'].nunique()} "
             f"runs ({int(short.sum())} partial bin(s) dropped)")
    else:
        note(f"{paths['timeseries']} not found: cof_time regime unavailable", logging.WARNING)

    return TriboData(runs=runs, checkpoints=checkpoints, cof_bins=cof_bins, inputs=inputs,
                     dropped_inputs=dropped, source=source, notes=notes)


# ---------------------------------------------------------------------------
def target_frame(data: TriboData, target: str | TargetSpec,
                 censor_mode: str = "exclude") -> tuple[pd.DataFrame, dict]:
    """Rows usable for ``target`` (inputs joined from the runs table) with ``y``, ``censored``,
    ``y_lod``; ``info`` counts excluded / censored / missing rows."""
    spec = TARGETS[target] if isinstance(target, str) else target
    info: dict = {"target": spec.name, "regime": spec.regime, "censor_mode": censor_mode}
    if spec.regime == "wear_checkpoints":
        src = data.checkpoints
    elif spec.regime == "cof_time":
        src = data.cof_bins
    else:
        src = data.runs
    if src is None or spec.column not in src:
        return pd.DataFrame(), {**info, "n_rows": 0, "n_runs": 0, "absent": True}
    if spec.regime in ROW_REGIMES:
        df = src.merge(data.runs.drop(columns=[c for c in src.columns
                                               if c in data.runs and c != "Experiment_ID"]),
                       on="Experiment_ID", how="inner")
    else:
        df = src.copy()
    val = pd.to_numeric(df[spec.column], errors="coerce").to_numpy(float)
    y = spec.forward(val)
    valid = np.isfinite(y)
    n_nonpos = int((np.isfinite(val) & ~valid).sum())
    censored = np.zeros(len(df), bool)
    y_lod = np.full(len(df), np.nan)
    if spec.censorable and spec.detected_column in df:
        not_det = (df[spec.detected_column] == False).fillna(False).to_numpy(bool)  # noqa: E712
        lod = pd.to_numeric(df[spec.lod_column], errors="coerce").to_numpy(float) \
            if spec.lod_column in df else np.full(len(df), np.nan)
        y_lod = spec.forward(lod)
        censored = not_det & ~valid
        info["n_censored"] = int(censored.sum())
        if censor_mode == "lod_half":
            imp = censored & np.isfinite(y_lod)
            y = np.where(imp, spec.forward(lod / 2.0), y)
            valid = valid | imp
        elif censor_mode != "exclude":
            raise ValueError(f"censoring mode must be 'exclude' or 'lod_half', got {censor_mode!r}")
        info["n_censored_excluded"] = int((censored & ~valid).sum())
    df["y"], df["censored"], df["y_lod"] = y, censored, y_lod
    info.update(n_nonpositive=n_nonpos, n_missing=int((~valid & ~censored).sum()))
    df = df[valid].reset_index(drop=True)
    info.update(n_rows=len(df), n_runs=df["Experiment_ID"].nunique())
    return df, info
