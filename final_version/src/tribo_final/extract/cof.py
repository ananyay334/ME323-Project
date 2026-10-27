"""Offset-free COF per reciprocating cycle.

The instrument column DAQ.COF (Fx) = |Fx|/Fz sample by sample is NOT used as the COF: it carries the sensor's
sideways offset in the static step, and its mean includes the reversal samples where Fx passes through zero
(about 12 % low, more at high frequency). Instead, per cycle:

    COF = (P90(Fx) - P10(Fx)) / 2 / mean(Fz)

which cancels any constant Fx offset (forward and return strokes are symmetric about it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from ..io.csv_reader import FrictionLog


@dataclass
class SlidingStep:
    mask: np.ndarray  # rows of the sliding step (incl. ramps)
    start: int
    stop: int  # exclusive
    method: str


def find_sliding_step(df: pd.DataFrame, cfg: dict) -> SlidingStep:
    """One continuous sliding step; the static loading step never counts."""
    n = len(df)
    if "freq" in df.columns and df["freq"].notna().any() and (df["freq"] > cfg["freq_threshold_hz"]).any():
        on = (df["freq"] > cfg["freq_threshold_hz"]).to_numpy()
        method = "frequency channel"
    else:
        on = np.zeros(n, bool)
        scores = {}
        for step, blk in df.groupby("recipe_step"):
            if len(blk) < cfg["min_block_rows"]:
                continue
            p10, p90 = np.percentile(blk["Fx"], [10, 90])
            scores[step] = (p90 - p10) / 2 / max(blk["Fz"].abs().mean(), 1e-9)
        best = max(scores.values())
        for step, sc in scores.items():
            if sc >= cfg["block_halfrange_frac"] * best:
                on |= (df["recipe_step"] == step).to_numpy()
        method = "RecipeStep block (Fx half-range)"
    # longest contiguous run of `on`
    idx = np.flatnonzero(on)
    splits = np.flatnonzero(np.diff(idx) > 1)
    segs = np.split(idx, splits + 1)
    seg = max(segs, key=len)
    mask = np.zeros(n, bool)
    mask[seg[0] : seg[-1] + 1] = True
    return SlidingStep(mask, int(seg[0]), int(seg[-1]) + 1, method)


def detect_reversals(fx: np.ndarray, fs: float, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fx sign changes with hysteresis around the offset-free centre.

    Returns (indices, directions(+1 up / -1 down), info).
    """
    w = max(1, int(round(cfg["median_window_s"] * fs)))
    w += 1 - w % 2  # odd, centred
    fxm = median_filter(np.asarray(fx, float), size=w, mode="nearest")
    p10, p90 = np.percentile(fxm, [10, 90])
    centre, half = (p10 + p90) / 2, (p90 - p10) / 2
    h = cfg["hysteresis_frac"] * half
    x = fxm - centre
    state = 0
    idx, direc = [], []
    for i, v in enumerate(x):
        if v > h:
            if state == -1:
                idx.append(i)
                direc.append(1)
            state = 1
        elif v < -h:
            if state == 1:
                idx.append(i)
                direc.append(-1)
            state = -1
    return np.array(idx, int), np.array(direc, int), {"centre": centre, "half_range": half, "hyst": h}


def extract_cof(log: FrictionLog, cfg_cof: dict, stroke_mm: float, nominal: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Per-cycle COF and the run-level COF targets for one friction log."""
    df = log.data
    fs = log.fs
    dt = 1.0 / fs
    step = find_sliding_step(df, cfg_cof)
    s = df.iloc[step.start : step.stop]
    t = s["t"].to_numpy()
    fx = s["Fx"].to_numpy()
    fz = s["Fz"].to_numpy()
    t0 = t[0]
    m_per_cycle = 4 * stroke_mm / 1000.0

    # cycles over the whole sliding step (ramps included)
    rev_all, dir_all, rinfo_all = detect_reversals(fx, fs, cfg_cof)
    if "freq" in s.columns and s["freq"].notna().all():
        cum_cycles = np.cumsum(s["freq"].clip(lower=0).to_numpy()) * dt
        n_cycles = float(cum_cycles[-1])
        cycles_method = "integrated frequency channel"
    else:
        cum_cycles = np.interp(np.arange(len(s)), np.r_[0, rev_all, len(s) - 1], np.r_[0, np.arange(1, len(rev_all) + 1) / 2, len(rev_all) / 2])
        n_cycles = len(rev_all) / 2
        cycles_method = "counted Fx reversals"

    # trimmed window: 1 s off each end (accel / decel ramps)
    trim = cfg_cof["ramp_trim_s"]
    tw0, tw1 = t[0] + trim, t[-1] - trim
    win = (t >= tw0) & (t <= tw1)
    wi = np.flatnonzero(win)
    a, b = wi[0], wi[-1] + 1
    rev, direc, rinfo = detect_reversals(fx[a:b], fs, cfg_cof)
    ups = rev[direc == 1] + a  # one cycle = upward switch -> next upward switch
    rows = []
    for k in range(len(ups) - 1):
        i0, i1 = ups[k], ups[k + 1]
        seg = fx[i0:i1]
        if len(seg) < 3:
            continue
        p10, p90 = np.percentile(seg, [10, 90])
        fzm = fz[i0:i1].mean()
        mid = (i0 + i1) // 2
        rows.append(
            {
                "cycle": k + 1,
                "t_s": t[mid] - t0,
                "s_m": cum_cycles[mid] * m_per_cycle,
                "cof": (p90 - p10) / 2 / fzm,
                "fx_offset": (p90 + p10) / 2,
                "Fz_mean": fzm,
                "n_samples": i1 - i0,
            }
        )
    cyc = pd.DataFrame(rows)
    tw_rel0, tw_rel1 = tw0 - t0, tw1 - t0

    res: dict = {}
    first = cyc[cyc.t_s <= tw_rel0 + cfg_cof["first_window_s"]]
    last = cyc[cyc.t_s >= tw_rel1 - cfg_cof["last_window_s"]]
    half = cyc.iloc[len(cyc) // 2 :]
    lo, hi = cfg_cof["dist_window_m"]
    dwin = cyc[(cyc.s_m >= lo) & (cyc.s_m <= hi)]
    res["COF_first10s"] = first.cof.mean()
    res["COF_end"] = half.cof.median()
    res["COF_last10s"] = last.cof.mean()
    res["COF_whole"] = cyc.cof.mean()
    res["COF_0.5_2m"] = dwin.cof.mean() if len(dwin) else np.nan
    if "cof_instr" in s.columns:
        res["COF_instrument_mean"] = s["cof_instr"].to_numpy()[a:b].mean()
    static = df.iloc[: step.start]
    static = static[static["recipe_step"] != s["recipe_step"].iloc[0]] if len(static) else static
    res["COF_instrument_static"] = static["cof_instr"].mean() if len(static) and "cof_instr" in static else np.nan
    res["Fx_static_N"] = static["Fx"].mean() if len(static) else np.nan

    # Fx offset and stroke asymmetry
    fxw = fx[a:b]
    centre, hyst = rinfo["centre"], rinfo["hyst"]
    res["Fx_offset_N"] = centre
    res["Fx_forward_N"] = float(np.median(fxw[fxw > centre + hyst]))
    res["Fx_return_N"] = float(np.median(fxw[fxw < centre - hyst]))
    res["Fz_mean_N"] = fz[a:b].mean()

    # cycle counting and its check against f * t
    res["cycles"] = n_cycles
    res["cycles_method"] = cycles_method
    res["cycles_reversals"] = len(rev_all) / 2
    res["sliding_time_s"] = (t[-1] - t[0]) + dt
    res["S_m"] = n_cycles * m_per_cycle
    res["n_cycles_window"] = len(cyc)
    res["sliding_method"] = step.method
    if nominal:
        ft = nominal["Freq_Hz"] * nominal["Run_time_s"]
        res["cycles_ratio_ft"] = n_cycles / ft
        res["qc_cycles_ok"] = abs(n_cycles / ft - 1) <= cfg_cof["cycle_check_tol"]
    # is the COF still rising at the end? slope over the second half of the cycles
    if len(half) > 5:
        slope = np.polyfit(half.t_s, half.cof, 1)[0]
        res["COF_slope_end_per_s"] = slope
        res["cof_rising_end"] = bool(slope > 0)
    res["t_window_start_s"] = tw_rel0
    res["t_window_end_s"] = tw_rel1
    return res, cyc


def measure_params(log: FrictionLog, cfg_cof: dict) -> dict:
    """Run parameters from the friction log itself (used only when no .lrx exists)."""
    df = log.data
    step = find_sliding_step(df, cfg_cof)
    s = df.iloc[step.start : step.stop]
    rev, _, _ = detect_reversals(s["Fx"].to_numpy(), log.fs, cfg_cof)
    dur = s["t"].iloc[-1] - s["t"].iloc[0] + 1.0 / log.fs
    static = df.iloc[: step.start]
    return {
        "static_s": float(len(static) / log.fs),
        "Load_N": round(s["Fz"].mean() * 2) / 2,
        "Freq_Hz": len(rev) / 2 / dur,
        "Run_time_s": float(round(dur)),
        "source": "csv",
    }
