"""Kinematics, contact mechanics and wear-rate formulas."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def distance_per_cycle_mm(stroke_mm: float, definition: str = "full") -> float:
    """Sliding distance of one reciprocating cycle (there and back)."""
    if definition == "full":
        return 2.0 * stroke_mm
    if definition == "amplitude":
        return 4.0 * stroke_mm
    raise ValueError(f"stroke_definition must be 'full' or 'amplitude', got {definition!r}")


def sliding_distance_m(freq_hz, stroke_mm, time_s, definition="full") -> float:
    """S = N_cycles * distance_per_cycle,  N_cycles = f * t   (DOE eq. 3)."""
    return freq_hz * time_s * distance_per_cycle_mm(stroke_mm, definition) / 1000.0


def mean_speed_mps(freq_hz, stroke_mm, definition="full") -> float:
    return freq_hz * distance_per_cycle_mm(stroke_mm, definition) / 1000.0


def hertz_contact(load_N, ball_d_mm, E1_GPa, nu1, E2_GPa, nu2) -> dict:
    """Ball-on-flat Hertz contact radius and pressures (initial, unworn)."""
    if not all(np.isfinite([load_N, ball_d_mm])) or load_N <= 0:
        return {"hertz_a_um": np.nan, "hertz_width_um": np.nan,
                "p_mean_MPa": np.nan, "p_max_MPa": np.nan}
    E_star = 1.0 / ((1 - nu1 ** 2) / (E1_GPa * 1e9) + (1 - nu2 ** 2) / (E2_GPa * 1e9))
    R = ball_d_mm / 2 / 1000.0
    a = (3 * load_N * R / (4 * E_star)) ** (1 / 3)
    p_mean = load_N / (math.pi * a ** 2)
    return {"hertz_a_um": a * 1e6, "hertz_width_um": 2 * a * 1e6,
            "p_mean_MPa": p_mean / 1e6, "p_max_MPa": 1.5 * p_mean / 1e6}


def specific_wear_rate(volume_mm3, load_N, distance_m) -> float:
    """k = V / (F * S)  in mm^3 / (N m)   (DOE eq. 2)."""
    if not (np.isfinite(volume_mm3) and np.isfinite(load_N) and np.isfinite(distance_m)):
        return float("nan")
    if load_N <= 0 or distance_m <= 0:
        return float("nan")
    return volume_mm3 / (load_N * distance_m)


def two_stage_wear(t, v_steady, dv_run, tau):
    """DOE eq. 4:  V(t) = V'_ss * t + dV_run * (1 - exp(-t / tau))."""
    return v_steady * t + dv_run * (1.0 - np.exp(-t / tau))


def fit_wear_trajectory(times_s, volumes_mm3) -> dict:
    """Linear slope through checkpoint volumes (>=2 points) and eq.-4 fit (>=4 points)."""
    t = np.asarray(times_s, float)
    v = np.asarray(volumes_mm3, float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    out = {"n_checkpoints": int(len(t))}
    if len(t) >= 2:
        slope, icpt = np.polyfit(t, v, 1)
        out["Vdot_linear_mm3_per_s"] = float(slope)
        pred = slope * t + icpt
        ss = ((v - v.mean()) ** 2).sum()
        out["Vdot_linear_R2"] = float(1 - ((v - pred) ** 2).sum() / ss) if ss > 0 else float("nan")
    if len(t) >= 4:
        try:
            p0 = [max(out.get("Vdot_linear_mm3_per_s", 1e-9), 1e-12), max(v.max() * 0.2, 1e-12),
                  max(t.max() * 0.1, 1.0)]
            popt, _ = curve_fit(two_stage_wear, t, v, p0=p0, bounds=([0, 0, 1e-3], [np.inf] * 3),
                                maxfev=20000)
            out.update({"Vdot_steady_mm3_per_s": float(popt[0]), "dV_run_mm3": float(popt[1]),
                        "tau_trans_s": float(popt[2])})
        except Exception as exc:     # noqa: BLE001
            out["trajectory_fit_error"] = str(exc)[:120]
    return out
