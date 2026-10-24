"""COF target extraction from the tribometer time series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .tribometer_csv import TriboRun


def identify_sliding_step(df: pd.DataFrame, setting="auto") -> int:
    """Return the RecipeStep that contains the reciprocating sliding.

    Approach / load-settling steps show near-zero COF, so among steps whose median
    COF is at least 30 % of the highest step median, the longest one is chosen.
    """
    steps = df["recipe_step"].dropna().unique()
    if setting not in (None, "auto", ""):
        s = int(float(setting))
        if s not in steps:
            raise ValueError(f"sliding_step={s} not present; steps in file: {sorted(steps)}")
        return s
    stats = []
    for s in steps:
        seg = df[df["recipe_step"] == s]
        stats.append((s, seg["t_s"].iloc[-1] - seg["t_s"].iloc[0], np.nanmedian(np.abs(seg["cof"]))))
    top = max(m for _, _, m in stats)
    cands = [(s, d) for s, d, m in stats if m >= 0.3 * top]
    return int(max(cands, key=lambda x: x[1])[0])


def _rolling(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(max(n, 1), center=True, min_periods=1).mean().to_numpy()


def extract_cof(run: TriboRun, cfg: dict, load_N: float | None = None) -> tuple[dict, pd.DataFrame, dict]:
    """Return (scalar targets, 1-Hz COF(t) series, internals for plotting)."""
    c = cfg["cof"]
    df = run.data
    step = identify_sliding_step(df, cfg["csv"].get("sliding_step", "auto"))
    seg = df[df["recipe_step"] == step].copy()
    seg["t"] = seg["t_s"] - seg["t_s"].iloc[0]
    if c.get("trim_end_s", 0) > 0:
        seg = seg[seg["t"] <= seg["t"].iloc[-1] - c["trim_end_s"]]
    t = seg["t"].to_numpy()
    mu = seg["cof"].to_numpy()
    T = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    dt = 1.0 / run.sample_rate_hz
    flags: list[str] = []

    # --- steady-state detection -------------------------------------------
    win = int(round(max(c["min_smooth_window_s"], c["smooth_window_frac"] * T) / dt))
    roll = _rolling(mu, win)
    last = t >= (1 - c["ref_last_frac"]) * T
    ref = float(np.nanmedian(mu[last]))
    mad = float(np.nanmedian(np.abs(roll[last] - np.nanmedian(roll[last])))) * 1.4826
    band = max(c["rel_tol"] * abs(ref), c["abs_tol"], 2 * mad)
    outside = np.abs(roll - ref) > band
    early = t < (1 - c["ref_last_frac"]) * T
    idx = np.where(outside & early)[0]
    t_ss = float(t[idx[-1]] + dt) if len(idx) else 0.0
    t_ss = max(t_ss, c["min_runin_frac"] * T)
    if t_ss > c["max_runin_frac"] * T:
        flags.append("long_running_in")
    if np.mean(outside[last]) > 0.2:
        flags.append("cof_unsteady_in_second_half")
    ss = t >= t_ss

    out = {
        "sliding_step": step,
        "sliding_time_s": round(T, 3),
        "sample_rate_hz": round(run.sample_rate_hz, 3),
        "n_samples_sliding": int(len(t)),
        "COF_ss_mean": float(np.nanmean(mu[ss])),
        "COF_ss_median": float(np.nanmedian(mu[ss])),
        "COF_ss_std": float(np.nanstd(mu[ss])),
        "COF_overall_mean": float(np.nanmean(mu)),
        "COF_runin_peak": float(np.nanmax(roll[~ss])) if (~ss).any() else float("nan"),
        "t_runin_s": round(t_ss, 3),
        "runin_frac": round(t_ss / T, 4) if T > 0 else float("nan"),
    }

    # --- checkpoints (for the COF(t) regime of the DOE) --------------------
    for cp in c.get("checkpoints_s", []):
        w = c.get("checkpoint_window_s", 5) / 2
        m = (t >= cp - w) & (t <= cp + w)
        out[f"COF_at_{int(cp)}s"] = float(np.nanmean(mu[m])) if m.any() else float("nan")

    # --- load QC ----------------------------------------------------------
    if "fz_N" in seg:
        fz = seg["fz_N"].to_numpy()
        out["Fz_mean_N"] = float(np.nanmean(fz))
        out["Fz_std_N"] = float(np.nanstd(fz))
        if load_N and np.isfinite(load_N):
            dev = abs(out["Fz_mean_N"] - load_N) / load_N
            out["Fz_dev_frac"] = round(dev, 4)
            if dev > c["load_tolerance_frac"]:
                flags.append("load_mismatch")

    # --- servo Z drift: diagnostic only (NOT used as wear) -----------------
    if "z_depth_um" in seg:
        z = seg["z_depth_um"].to_numpy()
        n5 = max(int(0.05 * len(z)), 1)
        out["Zdepth_drift_um"] = float(np.nanmedian(z[-n5:]) - np.nanmedian(z[:n5]))
        out["Zdepth_monotonic_rho"] = float(spearmanr(t, z)[0]) if len(z) > 3 else float("nan")

    # --- binned COF(t) series ----------------------------------------------
    b = c["bin_s"]
    bins = np.floor(t / b).astype(int)
    g = pd.DataFrame({"bin": bins, "t": t, "cof": mu,
                      "fz": seg["fz_N"].to_numpy() if "fz_N" in seg else np.nan,
                      "z": seg["z_depth_um"].to_numpy() if "z_depth_um" in seg else np.nan})
    ts = g.groupby("bin").agg(Time_s=("t", "max"), COF=("cof", "mean"), COF_std=("cof", "std"),
                              Fz_N=("fz", "mean"), Zdepth_um=("z", "mean"), n=("cof", "size"))
    ts["Time_s"] = (ts.index.to_numpy() + 1) * b          # end of bin = elapsed sliding time
    ts = ts[ts["n"] >= 0.5 * b / dt].drop(columns="n")    # drop incomplete last bin
    ts["Phase"] = np.where(ts["Time_s"] <= t_ss, "running_in", "steady")
    ts = ts.reset_index(drop=True)

    internals = {"t": t, "mu": mu, "roll": roll, "ref": ref, "band": band, "t_ss": t_ss,
                 "fz": seg.get("fz_N"), "z": seg.get("z_depth_um")}
    out["cof_flags"] = ";".join(flags)
    return out, ts, internals


# ---------------------------------------------------------------------------
# Per-cycle COF (304 SS DOE: two equal slide steps, 1000 Hz, ramps at every step)
# ---------------------------------------------------------------------------
def sliding_segments(df: pd.DataFrame) -> list[np.ndarray]:
    """Row indices of every sliding RecipeStep, in time order.

    A step = a contiguous block of one RecipeStep value. It is sliding when, in order of
    preference: the logged reciprocating frequency exceeds 0.5 Hz; or the Fx half-range
    ((P90 - P10)/2) / mean Fz is at least 30 % of the largest block's (a static loading
    step has a constant Fx, so its half-range is ~0 even when Fx carries an offset); or,
    without Fx, the median |COF| is at least 30 % of the highest block median.
    """
    rs = df["recipe_step"].to_numpy()
    seg = np.concatenate([[0], np.cumsum(rs[1:] != rs[:-1])])
    blocks = [np.flatnonzero(seg == s) for s in range(seg[-1] + 1)] if len(seg) else []
    if "freq_Hz" in df and np.nanmax(np.nan_to_num(df["freq_Hz"].to_numpy(float))) > 0:
        fr = df["freq_Hz"].to_numpy(float)
        return [b for b in blocks if np.nanmax(np.nan_to_num(fr[b])) > 0.5]
    if "fx_N" in df and df["fx_N"].notna().any():
        fx, fz = df["fx_N"].to_numpy(float), df["fz_N"].to_numpy(float) if "fz_N" in df else None
        score = []
        for b in blocks:
            if len(b) < 10:
                score.append(np.nan)
                continue
            hr = np.subtract(*np.nanpercentile(fx[b], [90, 10])) / 2
            score.append(hr / abs(np.nanmean(fz[b])) if fz is not None and np.nanmean(fz[b]) else hr)
    else:
        score = [float(np.nanmedian(np.abs(df["cof"].to_numpy()[b]))) for b in blocks]
    top = np.nanmax(score) if len(score) else np.nan
    return [b for b, m in zip(blocks, score) if np.isfinite(m) and m >= 0.3 * top]


def _hysteresis_state(x: np.ndarray, h: float) -> np.ndarray:
    """+1 / -1 state that flips only when x crosses +h / -h (0 before the first crossing)."""
    s = np.where(x > h, 1.0, np.where(x < -h, -1.0, np.nan))
    return pd.Series(s).ffill().fillna(0.0).to_numpy()


def _reversals(x: np.ndarray, fs: float, med_s: float, hyst: float) -> np.ndarray:
    """Sample indices where the moving-median signal reverses sign (with hysteresis)."""
    from scipy.ndimage import median_filter
    w = max(int(round(med_s * fs)), 1) | 1
    xm = median_filter(x, size=w, mode="nearest")
    p10, p90 = np.nanpercentile(xm, [10, 90])
    st = _hysteresis_state(xm - (p10 + p90) / 2, hyst * (p90 - p10) / 2)
    return np.flatnonzero((st[1:] != st[:-1]) & (st[:-1] != 0)) + 1


def _upward_crossings(p: np.ndarray, hyst: float) -> np.ndarray:
    p1, p99 = np.nanpercentile(p, [1, 99])
    st = _hysteresis_state(p - np.nanmedian(p), hyst * (p99 - p1) / 2)
    return np.flatnonzero((st[1:] > st[:-1]) & (st[:-1] != 0)) + 1


def extract_cof_cycles(run: TriboRun, cfg: dict, load_N: float | None = None,
                       freq_hz: float | None = None, stroke_mm: float | None = None):
    """Per-cycle COF and per-checkpoint summary of a two-step reciprocating run.

    Returns ``(checkpoints, cycles, internals)``: one dict per sliding step (= checkpoint
    1, 2), a DataFrame ``Step, Cycle, t_s, COF_c, Fz_N, Halfrange_N, Amp_mm`` (t_s = sliding
    time at the cycle's middle) and plotting internals incl. the run-level ``flags``.
    """
    c = cfg["cof"]
    df = run.data
    fs = run.sample_rate_hz
    stroke_mm = float(stroke_mm if stroke_mm is not None else cfg["test"]["stroke_mm"])
    lo_pct, hi_pct = c.get("halfrange_pct", [10, 90])
    has_fx = "fx_N" in df and df["fx_N"].notna().any()
    flags: list[str] = []
    if not has_fx:
        flags.append("cof_from_cof_column")
    segs = sliding_segments(df)
    t_all = df["t_s"].to_numpy(float)
    trim0, trim1 = c.get("trim_step_start_s", 1.0), c.get("trim_step_end_s", 1.0)
    if len(segs) > 2:
        flags.append(f"{len(segs)}_sliding_steps_first_two_used")
    if len(segs) == 1 and c.get("split_single_step", True):
        # one continuous sliding step (the campaign protocol): checkpoint 1 = first half of the
        # run (t_run/2), checkpoint 2 = second half; ramps exist only at the real start / end
        flags.append("single_step_split_at_half")
        idx = segs[0]
        t = t_all[idx] - t_all[idx[0]]
        half = (t[-1] + 1.0 / fs) / 2
        steps = [(idx[t < half], trim0, 0.0), (idx[t >= half], 0.0, trim1)]
    else:
        if len(segs) < 2:
            flags.append(f"only_{len(segs)}_sliding_step")
        steps = [(i, trim0, trim1) for i in segs[:2]]
    cyc_rows, cps, amps, t_slide = [], [], [], 0.0
    pos_ok = False
    is_vel = "veloc" in run.column_map.get("position_mm", "").lower()     # counts cycles, no amplitude
    for j, (idx, tr0, tr1) in enumerate(steps, start=1):
        t = t_all[idx] - t_all[idx[0]]
        dur = float(t[-1] + 1.0 / fs) if len(t) else 0.0
        fx = df["fx_N"].to_numpy(float)[idx] if has_fx else None
        mu = df["cof"].to_numpy(float)[idx]
        fz = df["fz_N"].to_numpy(float)[idx] if "fz_N" in df else np.full(len(idx), np.nan)
        pos = df["position_mm"].to_numpy(float)[idx] if "position_mm" in df else None
        if pos is not None:
            p1, p99 = np.nanpercentile(pos, [1, 99])
            pos_ok = np.isfinite(p99 - p1) and (p99 - p1) / 2 > 0.1 * stroke_mm
            if not pos_ok and "position_static" not in flags:
                flags.append("position_static")
        # --- cycle boundaries on the full step (counting includes the ramps) ---
        if pos is not None and pos_ok:
            ups = _upward_crossings(pos, c.get("reversal_hysteresis", 0.2))
            n_cyc, bounds = len(ups), ups
        else:
            sig = fx if has_fx else mu
            rev = _reversals(sig, fs, c.get("reversal_median_s", 0.02), c.get("reversal_hysteresis", 0.2))
            if len(rev) >= 4:
                n_cyc, bounds = len(rev) // 2, rev[::2]
            elif freq_hz and np.isfinite(freq_hz):
                n_cyc = int(round(freq_hz * dur))
                bounds = np.searchsorted(t, np.arange(n_cyc + 1) / freq_hz)
                flags.append("cycles_nominal")
            else:
                n_cyc, bounds = 0, np.array([], int)
                flags.append("cycles_not_found")
        # --- per-cycle COF on complete cycles inside the trimmed window ---
        t0, t1 = tr0, dur - tr1
        vals = []
        for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            if b <= a or t[a] < t0 or t[b - 1] > t1:
                continue
            fzc = float(np.nanmean(fz[a:b])) if np.isfinite(fz[a:b]).any() else (load_N or np.nan)
            if has_fx:
                p_lo, p_hi = np.nanpercentile(fx[a:b], [lo_pct, hi_pct])
                half = (p_hi - p_lo) / 2
                cof_c = half / fzc if fzc else np.nan
            else:
                cof_c = float(np.nanmedian(np.abs(mu[a:b])))
                half = cof_c * fzc
            amp = float(np.subtract(*np.nanpercentile(pos[a:b], [99, 1])) / 2) \
                if (pos is not None and pos_ok and not is_vel) else np.nan
            amps.append(amp)
            vals.append(cof_c)
            cyc_rows.append({"Step": j, "Cycle": k, "t_s": t_slide + float((t[a] + t[b - 1]) / 2),
                             "COF_c": float(cof_c), "Fz_N": fzc, "Halfrange_N": float(half), "Amp_mm": amp})
        vals = np.asarray(vals, float)
        last = vals[int(np.floor(len(vals) * (1 - c.get("steady_last_frac", 0.5)))):]
        cps.append({"Checkpoint": j, "Cycles_step": int(n_cyc), "Step_duration_s": dur,
                    "COF_steady": float(np.nanmedian(last)) if len(last) else np.nan,
                    "COF_std": float(np.nanstd(last, ddof=1)) if len(last) > 1 else np.nan,
                    "Freq_achieved_Hz": n_cyc / dur if dur > 0 else np.nan,
                    "Fz_mean_N": float(np.nanmean(fz[(t >= t0) & (t <= t1)])) if np.isfinite(fz).any() else np.nan,
                    "n_cycles_cof": int(len(vals)),
                    "steady_from_s": cyc_rows[-len(last)]["t_s"] if len(last) else np.nan,
                    "steady_to_s": t_slide + t1})
        if "z_depth_um" in df:
            z = df["z_depth_um"].to_numpy(float)[idx]
            n5 = max(int(0.05 * len(z)), 1)
            cps[-1]["Zdepth_drift_um"] = float(np.nanmedian(z[-n5:]) - np.nanmedian(z[:n5]))   # diagnostic only
        t_slide += dur
    amp = float(np.nanmedian(amps)) if np.isfinite(amps).any() else np.nan
    if not np.isfinite(amp):
        amp = stroke_mm
        flags.append("stroke_nominal")
    cum = 0
    for cp in cps:
        cum += cp["Cycles_step"]
        cp.update({"Cycles_cum": cum, "Stroke_amp_mm": amp, "Distance_m": 4 * amp * cum / 1000.0})
    cycles = pd.DataFrame(cyc_rows, columns=["Step", "Cycle", "t_s", "COF_c", "Fz_N", "Halfrange_N", "Amp_mm"])
    return cps, cycles, {"flags": flags, "sample_rate_hz": fs, "position_used": bool(pos_ok)}
