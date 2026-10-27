"""The TA's wear/COF method (external/ta_analyze_wear/analyze_wear.py) with its bugs fixed.

The original is kept unchanged in ``external/ta_analyze_wear/``. Same overall method: plane-levelled
height map, mean across-track profile, baseline from the outer 15 % of rows, groove = area below the
baseline between the zero crossings around the deepest point, volume = area x 10 mm, k = V / (F * S),
and the most-worn scan of each run. Fixes (numbers refer to the bug list in ``report.md``):

 1. Orientation: the original picks the axis from the *unlevelled* map, so plate tilt / waviness decides,
    and 25 of 92 scans were transposed (the "profile" then runs along the track). Here the track axis is
    known (plate x); ``track_axis="auto"`` decides only after levelling.
 2. Deepest point: searched only between the two shoulder bands, not inside the reference rows.
 3. Groove area: the original zeroes every point shallower than the cutoff (2.5 x baseline SD, >= 0.25 um),
    which deletes shallow grooves entirely (25 scans had area 0) and biases the rest. Here the full area
    below the baseline is integrated, and the cutoff is used only as a detection limit (depth < cutoff:
    below LOD).
 4. Pile-up: the original sums everything above the baseline over the whole image outside the groove
    (including the unworn texture). Here only one groove width on each side of the groove counts.
 5. Validity: a groove reaching into the shoulder bands (track wider than the central 70 %, e.g. the
    89-100 N tracks) has no unworn reference; flagged ``one_sided`` and not used.
 6. Scan selection: the most-worn scan = largest groove area among valid, detected scans (the original uses
    the deepest point, which on these rough plates is often a texture valley, and includes invalid scans).
 7. COF: the original averages the instrument's |Fx|/Fz column per second. Reversal samples (Fx passing
    zero) pull that average down by ~12 %, more at high frequency (correlation with f: -0.68). Here COF =
    (P90 - P10)/2 of Fx / mean Fz per second (offset-free), and the last second (deceleration) is trimmed
    as well as the first.
 8. Inputs: load, frequency and run time come from the run's .lrx recipe (the original uses hard-coded
    defaults for R01). The friction CSV is ``<run>.csv`` (the original takes the first *.csv, which after
    one run can be its own output file), and nothing is written into the data folder.

    python -m tribo_extract.ta_wear            # -> data/processed/ta_wear_runs.csv, ta_wear_scans.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import PROJECT_ROOT, load_config
from .height_io import read_bcrf
from .recipe import recipe_params
from .tribometer_csv import read_tribometer_csv

AMP_MM = 5.0
TRACK_MM = 10.0
SHOULDER = 0.15


def level_plane(z: np.ndarray, shoulder: float = SHOULDER) -> tuple[np.ndarray, np.ndarray]:
    """Plane fitted to the outer ``shoulder`` rows (3 passes of 2-sigma clipping), as in the original."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.stack([np.ones_like(xx, float), (yy - ny / 2) / ny, (xx - nx / 2) / nx], axis=-1)
    e = int(ny * shoulder)
    mask = np.zeros((ny, nx), bool)
    mask[:e], mask[-e:] = True, True
    mask &= np.isfinite(z)
    for _ in range(3):
        c, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        res = z - A @ c
        sd = np.nanstd(res[mask])
        mask &= np.abs(res - np.nanmedian(res[mask])) < 2 * sd
    c, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
    return z - A @ c, mask


def orient(z: np.ndarray, track_axis: str = "x") -> np.ndarray:
    """Rows across the track. ``x``: track along plate x (rows = y), as in all our scans."""
    if track_axis == "x":
        return z
    if track_axis == "y":
        return z.T
    zl, _ = level_plane(z)                       # auto: decide on the levelled map
    return z.T if np.nanstd(np.nanmean(zl, axis=0)) > np.nanstd(np.nanmean(zl, axis=1)) else z


def groove_metrics(zl: np.ndarray, px_um: float, shoulder: float = SHOULDER) -> dict:
    prof = np.nanmean(zl, axis=1)
    w = min(51, len(prof) // 4 * 2 + 1)
    if w > 5:
        prof = savgol_filter(prof, window_length=w, polyorder=3)
    n = len(prof)
    e = int(n * shoulder)
    sh = np.r_[0:e, n - e:n]
    base, base_sd = float(np.median(prof[sh])), float(np.std(prof[sh]))
    d = base - prof                                                   # + = below baseline
    k = e + int(np.argmax(d[e:n - e]))                                # deepest point between the shoulders
    dmax = float(d[k])
    left = next((i for i in range(k, -1, -1) if d[i] <= 0), 0)
    right = next((i for i in range(k, n) if d[i] <= 0), n - 1)
    width = (right - left) * px_um
    area = float(np.clip(d[left:right + 1], 0, None).sum() * px_um)
    side = max(right - left, 1)
    pile = float(np.clip(-d[max(0, left - side):left], 0, None).sum() * px_um
                 + np.clip(-d[right + 1:min(n, right + 1 + side)], 0, None).sum() * px_um)
    cutoff = max(2.5 * base_sd, 0.25)
    one_sided = left < e or right > n - 1 - e
    return {"depth_max_um": dmax, "width_um": width, "area_groove_um2": area, "area_pileup_um2": pile,
            "cutoff_um": cutoff, "detected": dmax >= cutoff, "one_sided": bool(one_sided),
            "baseline_sd_um": base_sd, "vol_mm3": area * 1e-6 * TRACK_MM}


def friction(csv: Path, cfg: dict) -> dict:
    """Per-second (P90-P10)/2 of Fx / mean Fz over the sliding step, first and last second trimmed."""
    run = read_tribometer_csv(csv, cfg["csv"]["columns"])
    d = run.data
    step = d["recipe_step"].value_counts().idxmax()                   # the sliding step is the longest
    s = d[d["recipe_step"] == step]
    t = s["t_s"] - s["t_s"].iloc[0]
    s = s[(t > 1.0) & (t < t.max() - 1.0)]
    sec = ((s["t_s"] - s["t_s"].iloc[0]) // 1).astype(int)
    g = s.groupby(sec)
    cof = (g["fx_N"].quantile(0.9) - g["fx_N"].quantile(0.1)) / 2 / g["fz_N"].mean()
    return {"cof_initial": float(cof.iloc[:10].mean()), "cof_steady_state": float(cof.iloc[-10:].mean()),
            "cof_overall_mean": float(cof.mean()), "Fz_mean": float(s["fz_N"].mean())}


def analyse_run(run_dir: Path, cfg: dict, track_axis: str = "x") -> tuple[dict, pd.DataFrame]:
    rid = run_dir.name
    p = recipe_params(next(run_dir.glob("*.lrx")))
    L, f, t = p["Load_N"], p["Freq_Hz"], p["Run_time_s"]
    S = 4 * AMP_MM / 1000 * f * t
    rows = []
    for fp in sorted(run_dir.glob("*.bcrf"), key=lambda x: [int(c) if c.isdigit() else c
                                                             for c in re.split(r"(\d+)", x.name)]):
        hm = read_bcrf(fp)
        z = np.where(np.isfinite(hm.z), hm.z, np.nanmedian(hm.z))
        zl, _ = level_plane(orient(z, track_axis))
        m = groove_metrics(zl, hm.dx_um)
        m["file"] = fp.name
        rows.append(m)
    sc = pd.DataFrame(rows)
    sc.insert(0, "Experiment_ID", rid)
    ok = sc[~sc["one_sided"]]
    det = ok[ok["detected"]]
    out = {"Experiment_ID": rid, "Load_N": L, "Freq_Hz": f, "Run_time_s": t, "S_m": S,
           "n_scans": len(sc), "n_valid": len(ok), "n_detected": len(det)}
    if len(det):
        top = det.loc[det["area_groove_um2"].idxmax()]
        out.update({"selected": top["file"], "status": "detected", "depth_um": top["depth_max_um"],
                    "width_um": top["width_um"], "area_um2": top["area_groove_um2"], "vol_mm3": top["vol_mm3"],
                    "k": top["vol_mm3"] / (L * S)})
    else:
        out.update({"selected": "", "status": "below_LOD" if len(ok) else "not_measurable", "depth_um": np.nan,
                    "width_um": np.nan, "area_um2": np.nan, "vol_mm3": np.nan, "k": np.nan})
    out.update(friction(run_dir / f"{rid}.csv", cfg))
    return out, sc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="TA wear method, bugs fixed")
    ap.add_argument("--raw", default=str(PROJECT_ROOT / "testdata"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed"))
    ap.add_argument("--track-axis", default="x", choices=["x", "y", "auto"])
    a = ap.parse_args(argv)
    cfg = load_config()
    runs, scans = [], []
    for d in sorted(Path(a.raw).iterdir()):
        if d.is_dir() and re.match(r"^[RT]\d{2}$", d.name) and any(d.glob("*.bcrf")):
            r, s = analyse_run(d, cfg, a.track_axis)
            runs.append(r)
            scans.append(s)
            print(f"{r['Experiment_ID']}: {r['status']:>14}  k={r['k']:.3g}  COF={r['cof_steady_state']:.4f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).to_csv(Path(a.out) / "ta_wear_runs.csv", index=False)
    pd.concat(scans).to_csv(Path(a.out) / "ta_wear_scans.csv", index=False)


if __name__ == "__main__":
    main()
