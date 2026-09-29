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
