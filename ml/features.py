"""Feature sets, derived features and leakage guards.

* ``base``    : Carbon_pct, Hardness_HV, Load_N, Freq_Hz
* ``physics`` : base + Speed_mps, Hertz_p_mean_MPa, PV (= p_mean x v), plus
  Initial_Ra_um / Ambient_T_C / RH_pct when they are logged for enough runs.
* COF(t) regime: the run features + a time encoding (Time_s and/or log1p(Time_s)).

With the stroke fixed, Speed_mps is an exact multiple of Freq_Hz and Hertz_p_mean_MPa
a function of Load_N only, so the physics set adds no new design information; it
changes the *representation* (e.g. the Hertz term is a concave transform of load).
Plain OLS drops exactly collinear columns inside each training fold
(:class:`ml.models.linear.DropCollinear`); every other model is regularised.
"""
from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd

from tribo_extract import physics
from tribo_extract.config import load_config as load_extraction_config

from .data import TARGETS, TriboData

log = logging.getLogger("ml.features")

DERIVED: dict[str, tuple[str, ...]] = {"PV": ("Hertz_p_mean_MPa", "Speed_mps")}
TIME_ENCODINGS = {"raw": ["Time_s"], "log1p": ["log1p_Time_s"], "both": ["Time_s", "log1p_Time_s"]}
DESIGN = ["Carbon_pct", "Load_N", "Freq_Hz"]

# Never model inputs: identifiers, the servo-Z diagnostic, and every target column.
NEVER_INPUTS = ({"Zdepth_um", "Experiment_ID", "group", "is_holdout", "y", "censored", "y_lod",
                 "status", "Replicate_of", "Wear_detected", "Phase", "COF_std"}
                | {t.column for t in TARGETS.values()}
                | {t.lod_column for t in TARGETS.values() if t.lod_column})
# Per-run targets: time is an endpoint, not an input (also covers duration proxies).
RUN_REGIME_FORBIDDEN = {"Time_s", "log1p_Time_s", "sliding_time_s", "Sliding_distance_m",
                        "N_cycles"}


_LOGGED: set[str] = set()


def _log_once(msg: str, level: int = logging.INFO) -> None:
    if msg not in _LOGGED:
        _LOGGED.add(msg)
        log.log(level, msg)


def check_inputs(columns: list[str], regime: str) -> None:
    """Raise if a feature list contains a forbidden column for this regime."""
    bad = set(columns) & NEVER_INPUTS
    if regime == "run":
        bad |= set(columns) & RUN_REGIME_FORBIDDEN
    if bad:
        raise ValueError(f"forbidden input column(s) for the {regime} regime: {sorted(bad)}")


def resolve_feature_set(name: str, data: TriboData, cfg: dict) -> list[str]:
    """Columns of feature set ``name`` that the data can supply (derived ones included)."""
    fcfg = cfg["features"]
    if name not in fcfg["sets"]:
        raise KeyError(f"unknown feature set {name!r}; defined: {list(fcfg['sets'])}")
    cols = list(fcfg["sets"][name])
    if name == "physics":
        cv = data.cv_runs
        thr = fcfg.get("optional_min_coverage", 0.8)
        for c in fcfg.get("optional_physics", []):
            if c in data.inputs and c not in cols:
                cov = float(cv[c].notna().mean())
                if cov >= thr:
                    cols.append(c)
                else:
                    _log_once(f"feature {c} logged for {100 * cov:.0f} % of runs "
                              f"(< {100 * thr:.0f} %): not used")
    out = []
    for c in cols:
        need = DERIVED.get(c, (c,))
        if all(n in data.inputs for n in need):
            out.append(c)
        else:
            _log_once(f"feature set {name}: {c} unavailable (needs {need}) -> skipped",
                      logging.WARNING)
    check_inputs(out, "run")
    return out


def time_columns(cfg: dict) -> list[str]:
    enc = cfg["features"].get("time_encoding", "both")
    if enc not in TIME_ENCODINGS:
        raise ValueError(f"time_encoding must be one of {list(TIME_ENCODINGS)}, got {enc!r}")
    return TIME_ENCODINGS[enc]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (PV, log1p time) to a copy of ``df`` where the inputs exist."""
    df = df.copy()
    if {"Hertz_p_mean_MPa", "Speed_mps"} <= set(df.columns):
        df["PV"] = df["Hertz_p_mean_MPa"] * df["Speed_mps"]
    if "Time_s" in df:
        df["log1p_Time_s"] = np.log1p(df["Time_s"].astype(float))
    return df


def build_X(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Model matrix with exactly ``columns`` (float), derived features computed on the fly."""
    d = add_derived(df)
    return d[columns].astype(float).reset_index(drop=True)


# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _contact_constants() -> dict:
    t = load_extraction_config()["test"]
    return {k: t[k] for k in ("stroke_mm", "stroke_definition", "ball_diameter_mm", "E_ball_GPa",
                              "nu_ball", "E_flat_GPa", "nu_flat")}


def complete_inputs(primary: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Fill every input column physically consistently from Carbon/Load/Freq.

    Used wherever inputs are *set* rather than measured (partial dependence, GPR maps,
    active learning): Speed and Hertz pressure are recomputed with
    :mod:`tribo_extract.physics` (stroke from the reference runs, constants from
    ``config/extraction.yaml``); hardness, if not given, is the reference mean of that
    carbon level (linearly interpolated between levels); any other input is held at
    the reference median.
    """
    k = _contact_constants()
    out = primary.copy().reset_index(drop=True)
    stroke = float(reference["Stroke_mm"].median()) if "Stroke_mm" in reference else k["stroke_mm"]
    if "Hardness_HV" in reference and "Hardness_HV" not in out:
        hv = reference.groupby("Carbon_pct")["Hardness_HV"].mean().dropna()
        out["Hardness_HV"] = (np.interp(out["Carbon_pct"], hv.index.to_numpy(float), hv.to_numpy())
                              if len(hv) else np.nan)
    out["Stroke_mm"] = stroke
    out["Speed_mps"] = [physics.mean_speed_mps(f, stroke, k["stroke_definition"])
                        for f in out["Freq_Hz"]]
    out["Hertz_p_mean_MPa"] = [physics.hertz_contact(L, k["ball_diameter_mm"], k["E_ball_GPa"],
                                                     k["nu_ball"], k["E_flat_GPa"],
                                                     k["nu_flat"])["p_mean_MPa"]
                               for L in out["Load_N"]]
    for c in reference.columns:
        if c not in out and pd.api.types.is_numeric_dtype(reference[c]):
            out[c] = reference[c].median()
    return add_derived(out)


def check_physics_consistency(runs: pd.DataFrame, tol: float = 0.02) -> list[str]:
    """Compare the data's Speed/Hertz columns with a recomputation (catches config drift)."""
    msgs = []
    sub = runs.dropna(subset=["Load_N", "Freq_Hz"])
    if sub.empty:
        return msgs
    re = complete_inputs(sub[DESIGN + (["Hardness_HV"] if "Hardness_HV" in sub else [])], sub)
    for c in ("Speed_mps", "Hertz_p_mean_MPa"):
        if c in sub:
            a, b = sub[c].to_numpy(float), re[c].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
            if ok.any() and np.nanmax(np.abs(a[ok] / b[ok] - 1)) > tol:
                msgs.append(f"{c} in the data differs from tribo_extract.physics by up to "
                            f"{100 * np.nanmax(np.abs(a[ok] / b[ok] - 1)):.0f} % (stroke "
                            "definition / constants changed?) - partial-dependence and map "
                            "inputs use the recomputed values")
    return msgs
