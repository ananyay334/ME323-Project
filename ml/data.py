"""Load the run x checkpoint table (``runs_checkpoints.csv``) for modelling.

* rows: one per run x checkpoint; ``status != "error"`` and ``Include == "Y"`` (when the
  column exists) only;
* ``split="train"``: roles completed / corner / interior / center_replicate / topup
  (R00-R23 + top-ups); ``split="test"``: role ``test`` (R24-R29). Test rows are dropped
  right after the file is read unless ``split="test"``, which only ``ml.train --final`` asks for;
* inputs: ln(Load_N), ln(Freq_Hz), ln(Time_s) from the nominal values (standardised inside
  the estimators, i.e. on training rows only);
* targets: log10(k_mm3_per_Nm) and COF_steady. A row missing one target is kept for the other
  (masked as NaN). ``Wear_detected == False`` is censoring (< LOD), never zero: excluded for
  log10 k by default, or imputed at LOD/2 with ``censoring.mode: lod_half``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import data_paths
from .splits import assign_groups

log = logging.getLogger("ml.data")

INPUTS = {"ln_L": "Load_N", "ln_f": "Freq_Hz", "ln_t": "Time_s"}
TARGETS = {"log10_k": "log10 k [mm³/(N·m)]", "COF_steady": "COF_steady (per-cycle median)",
           "log10_w": "log10 track width [µm]"}
META = ["Experiment_ID", "group", "Role", "Replicate_of", "Checkpoint", "Load_N", "Freq_Hz",
        "Run_time_s", "Time_s"]


def parse_bool(s: pd.Series) -> pd.Series:
    """'True'/'False'/1/0/bool -> boolean with NA for anything else."""
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False,
         "yes": True, "no": False}
    return s.map(lambda v: v if isinstance(v, (bool, np.bool_)) else m.get(str(v).strip().lower(),
                                                                          pd.NA)).astype("boolean")


@dataclass
class TriboData:
    """Model-ready rows: meta columns, inputs ``X`` and the target matrix ``Y`` (NaN = masked)."""
    frame: pd.DataFrame
    targets: list[str]
    split: str
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[list(INPUTS)].astype(float).reset_index(drop=True)

    @property
    def Y(self) -> np.ndarray:
        return self.frame[self.targets].to_numpy(float)

    @property
    def groups(self) -> np.ndarray:
        return self.frame["group"].astype(str).to_numpy()

    @property
    def censored(self) -> np.ndarray:
        return self.frame[[f"censored_{t}" for t in self.targets]].to_numpy(bool)

    def note(self, msg: str, level: int = logging.INFO) -> None:
        self.notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)


def load_data(cfg: dict, source: str, split: str = "train") -> TriboData:
    """Rows of ``split`` ('train' | 'test') from ``source`` ('real' | 'synthetic')."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = data_paths(cfg, source)["checkpoints"]
    dcfg = cfg["data"]
    notes: list[str] = []

    def note(msg, level=logging.INFO):
        notes.append(("⚠ " if level >= logging.WARNING else "") + msg)
        log.log(level, msg)

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. For real data run `python -m tribo_extract extract`; for "
            "synthetic data run `python -m ml.synthetic` and pass `--data synthetic`.")
    df = pd.read_csv(path, dtype={"Experiment_ID": str, "Replicate_of": str, "Role": str, "status": str})
    roles = dcfg["train_roles"] if split == "train" else dcfg["test_roles"]
    role = df.get("Role", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    df = df[role.isin([r.lower() for r in roles])].copy()          # test seal: other split dropped here
    n0 = len(df)
    df["Experiment_ID"] = df["Experiment_ID"].astype(str).str.strip()
    status = df.get("status", pd.Series("ok", index=df.index)).astype(str).str.strip().str.lower()
    bad = status.isin([s.lower() for s in dcfg["drop_status"]])
    if "Include" in df:
        bad |= df["Include"].astype(str).str.strip().str.upper().eq("N")
    if bad.any():
        note(f"dropped {int(bad.sum())} row(s) (status error / Include = N): "
             f"{sorted(set(df.loc[bad, 'Experiment_ID']))}")
    df = df[~bad].copy()
    if df.empty:
        raise ValueError(f"no usable {split} rows in {path} ({n0} {split} row(s) before cleaning). Run "
                         "`python -m tribo_extract extract` on the campaign data first, or use "
                         "`--data synthetic`.")
    for c in ["Load_N", "Freq_Hz", "Time_s", "Run_time_s", "Checkpoint", "k_mm3_per_Nm", "COF_steady",
              "Wear_Volume_LOD_mm3", "Distance_m", "Track_width_um"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    nodesign = df[list(INPUTS.values())].le(0).any(axis=1) | df[list(INPUTS.values())].isna().any(axis=1)
    if nodesign.any():
        note(f"dropped {int(nodesign.sum())} row(s) without positive Load_N / Freq_Hz / Time_s",
             logging.WARNING)
        df = df[~nodesign].copy()
    for k, col in INPUTS.items():
        df[k] = np.log(df[col].astype(float))
    if "Replicate_of" not in df:
        df["Replicate_of"] = np.nan
    df["Replicate_of"] = df["Replicate_of"].map(
        lambda v: v.strip() if isinstance(v, str) and v.strip() not in ("", "nan") else np.nan)
    df["group"] = assign_groups(df)

    # targets
    det = parse_bool(df["Wear_detected"]) if "Wear_detected" in df else pd.Series(pd.NA, index=df.index,
                                                                                    dtype="boolean")
    k = df["k_mm3_per_Nm"].to_numpy(float)
    k_lod = (df["Wear_Volume_LOD_mm3"] / (df["Load_N"] * df["Distance_m"])).to_numpy(float)
    cens = (det == False).fillna(False).to_numpy(bool) & ~(k > 0)              # noqa: E712
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log10_k"] = np.where(k > 0, np.log10(np.where(k > 0, k, 1.0)), np.nan)
        df["log10_k_lod"] = np.where(k_lod > 0, np.log10(np.where(k_lod > 0, k_lod, 1.0)), np.nan)
    mode = cfg["censoring"]["mode"]
    if mode == "lod_half":
        imp = cens & np.isfinite(df["log10_k_lod"].to_numpy(float))
        df.loc[imp, "log10_k"] = df.loc[imp, "log10_k_lod"] - np.log10(2.0)
    elif mode != "exclude":
        raise ValueError(f"censoring.mode must be 'exclude' or 'lod_half', got {mode!r}")
    df["censored_log10_k"] = cens
    df["censored_COF_steady"] = False
    w = df["Track_width_um"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log10_w"] = np.where(w > 0, np.log10(np.where(w > 0, w, 1.0)), np.nan)   # robust wear measure
    df["censored_log10_w"] = False
    targets = [t for t in cfg.get("targets", list(TARGETS)) if t in TARGETS]
    keep = df[targets].notna().any(axis=1)
    if (~keep).any():
        note(f"dropped {int((~keep).sum())} row(s) with no target value: "
             f"{sorted(set(df.loc[~keep, 'Experiment_ID']))}")
    df = df[keep].sort_values(["Experiment_ID", "Checkpoint"]).reset_index(drop=True)
    note(f"{split}: {len(df)} rows from {df['Experiment_ID'].nunique()} runs in {df['group'].nunique()} "
         f"groups; " + "; ".join(f"{t}: {int(df[t].notna().sum())} rows" for t in targets)
         + f" ({int(cens.sum())} log10_k censored, mode {mode})")
    cols = META + list(INPUTS) + targets + [f"censored_{t}" for t in targets] + ["log10_k_lod"]
    return TriboData(frame=df[[c for c in cols if c in df]], targets=targets, split=split, source=source,
                     notes=notes)


def load_cof_cycles(cfg: dict, source: str, ids: list[str]) -> pd.DataFrame | None:
    """Per-cycle COF of the runs ``ids`` (for the replicate overlay figure)."""
    p = data_paths(cfg, source)["cof_cycles"]
    if not p.exists():
        return None
    c = pd.read_csv(p, dtype={"Experiment_ID": str})
    c = c[c["Experiment_ID"].isin(ids)]
    return c if len(c) else None
