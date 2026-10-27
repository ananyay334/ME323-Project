"""C6 - The TA's analyze_wear.py.

1. Its functions are run UNCHANGED (imported from inputs/analyze_wear.py) on all runs, in the same order as its
   run_pipeline(), but nothing is written into testdata/ (run_pipeline itself writes CSVs into the data folder).
2. A fixed copy (below) corrects the known bugs:
   1 orientation chosen from the unlevelled map (tilt decides)      -> rows always along the track
   2 points shallower than the cutoff zeroed one by one              -> integrate the whole groove; cutoff = detection only
   3 pile-up summed over the whole image outside the groove          -> only on the groove flanks (one width each side)
   4 shoulder (reference) rows never checked to be unworn            -> drop shoulder rows inside the scar, need 110 um/side
   5 scan picked by its deepest point (texture valleys)              -> largest groove area among valid scans
   6 COF = mean of the instrument |Fx|/Fz channel, decel not trimmed -> per-second half-range (P90-P10)/2/Fz, both ramps trimmed
   7 hard-coded R01 parameters, first *.csv, outputs into data folder -> .lrx parameters, the run's own log, nothing written
3. Both are compared with our pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

from ..extract.wear_band import fit_plane
from ..utils import quiet_warnings

TRACK_MM = 10.0


def load_ta_module(path: Path):
    import matplotlib

    matplotlib.use("Agg")
    name = "ta_analyze_wear"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


# --------------------------------------------------------------------------------------------- fixed copy
def fixed_groove_metrics(zl: np.ndarray, px: float, shoulder: np.ndarray) -> dict:
    """The TA's compute_groove_metrics with bugs 2 and 3 fixed; `shoulder` = boolean rows used as baseline."""
    prof = np.nanmean(zl, axis=1)
    wl = min(51, len(prof) // 4 * 2 + 1)
    if wl > 5:
        prof = savgol_filter(prof, window_length=wl, polyorder=3)
    base = np.median(prof[shoulder])
    base_sd = np.std(prof[shoulder])
    d = base - prof
    k = int(np.nanargmax(d))
    L = k
    while L > 0 and d[L] > 0:
        L -= 1
    R = k
    while R < len(d) - 1 and d[R] > 0:
        R += 1
    cutoff = max(2.5 * base_sd, 0.25)
    groove = float(np.clip(d[L : R + 1], 0, None).sum() * px)  # fix 2: no point-wise zeroing
    w = R - L
    pile = float(np.clip(-d[max(0, L - w) : L], 0, None).sum() * px + np.clip(-d[R + 1 : R + 1 + w], 0, None).sum() * px)
    return {"depth_max_um": float(d[k]), "width_um": w * px, "area_groove_um2": groove, "area_pileup_um2": pile,
            "cutoff_um": cutoff, "detected": bool(d[k] >= cutoff and groove > 0), "edges": (L, R), "baseline_sd_um": base_sd}


def fixed_scan(ta, z: np.ndarray, px: float, margin_um: float = 10.0, min_unworn_um: float = 110.0) -> dict:
    zc = ta.inpaint_voids(z)  # fix 1: never transpose - rows run along the track
    zl, mask = ta.robust_level(zc, order=1, shoulder_fraction=0.15)
    ny = zl.shape[0]
    shoulder = np.zeros(ny, bool)
    e = int(ny * 0.15)
    shoulder[:e] = shoulder[-e:] = True
    m = fixed_groove_metrics(zl, px, shoulder)
    # fix 4: shoulder rows must be unworn - drop rows inside the scar (+ margin), relevel, check both sides
    L, R = m["edges"]
    mg = int(np.ceil(margin_um / px))
    out = np.ones(ny, bool)
    out[max(0, L - mg) : min(ny, R + mg + 1)] = False
    left, right = out[: max(0, L - mg)].sum() * px, out[min(ny, R + mg + 1) :].sum() * px
    valid = left >= min_unworn_um and right >= min_unworn_um
    if out.sum() > 10:
        plane, _ = fit_plane(zc, np.broadcast_to(out[:, None], zc.shape), clip_sigma=2.0, passes=3)
        m = fixed_groove_metrics(zc - plane, px, out)
    m.update(valid=bool(valid), unworn_left_um=left, unworn_right_um=right)
    m["vol_track_mm3"] = m["area_groove_um2"] * 1e-6 * TRACK_MM
    return m


def fixed_cof(df: pd.DataFrame, trim_s: float = 1.0) -> float:
    """Fix 6: per-second half-range of Fx over Fz, both ramps trimmed."""
    s = df[df["step"] == "2"].copy() if "2" in set(df["step"]) else df.copy()
    s = s[(s.t > s.t.min() + trim_s) & (s.t < s.t.max() - trim_s)]
    sec = (s.t // 1).astype(int)
    hr = s.groupby(sec).apply(lambda g: (g.Fx.quantile(0.9) - g.Fx.quantile(0.1)) / 2 / g.Fz.mean())
    return float(hr.mean())


# --------------------------------------------------------------------------------------------- per scan / run
def _scan_job(ta_path: str, run_id: str, bcrf: str) -> dict:
    quiet_warnings()
    ta = load_ta_module(Path(ta_path))
    z, px, _ = ta.read_bcrf(bcrf)
    zo = ta.orient_across_track(z)
    flipped = zo.shape != z.shape
    zl, mask = ta.robust_level(ta.inpaint_voids(zo), order=1, shoulder_fraction=0.15)
    m, _, _ = ta.compute_groove_metrics(zl, px, mask, track_len_mm=TRACK_MM)
    zl2, mask2 = ta.robust_level(ta.inpaint_voids(z), order=1, shoulder_fraction=0.15)
    m2, _, _ = ta.compute_groove_metrics(zl2, px, mask2, track_len_mm=TRACK_MM)
    fx = fixed_scan(ta, z, px)
    row = {"run_id": run_id, "file": Path(bcrf).name, "flipped": flipped}
    row.update({f"orig_{k}": v for k, v in m.items()})
    row.update({f"axisfix_{k}": v for k, v in m2.items()})
    row.update({f"fixed_{k}": v for k, v in fx.items() if k != "edges"})
    row["fixed_edge_L"], row["fixed_edge_R"] = fx["edges"]
    return row


def _cof_job(ta_path: str, run_id: str, csv: str) -> dict:
    quiet_warnings()
    ta = load_ta_module(Path(ta_path))
    df = ta.read_friction_csv(csv)
    fric, _ = ta.compute_friction_features(df)
    return {"run_id": run_id, **{f"orig_{k}": v for k, v in fric.items()}, "fixed_cof": fixed_cof(df)}


def run_c6(cfg: dict, paths, runs: pd.DataFrame, per_loc: pd.DataFrame, n_jobs: int):
    from ..io.layout import find_runs

    ta_path = str(paths.inputs / "analyze_wear.py")
    folders = {r.folder_id: r for r in find_runs(paths.testdata)}
    fid = dict(zip(runs.run_id, runs.folder_id))
    scan_jobs = [(rid, str(s.bcrf)) for rid in runs.run_id for s in folders[fid[rid]].scans]
    scans = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_scan_job)(ta_path, r, b) for r, b in scan_jobs))
    cofs = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(_cof_job)(ta_path, rid, str(folders[fid[rid]].csv))
                                                for rid in runs.run_id))
    A = cfg["experiment"]["stroke_amplitude_mm"]
    rows = []
    for _, r in runs.iterrows():
        sc = scans[scans.run_id == r.run_id]
        S_nom = 4 * A / 1000 * r.Freq_Hz * r.Run_time_s
        o = sc.loc[sc.orig_depth_max_um.idxmax()]  # bug 5 as written: deepest point
        row = {"run_id": r.run_id, "Load_N": r.Load_N, "Freq_Hz": r.Freq_Hz, "S_nominal_m": S_nom,
               "orig_scan": o.file, "orig_depth_um": o.orig_depth_max_um, "orig_area_um2": o.orig_area_groove_um2,
               "orig_V_mm3": o.orig_vol_track_mm3, "orig_k": o.orig_vol_track_mm3 / (r.Load_N * S_nom),
               "orig_k_R01_defaults": o.orig_vol_track_mm3 / (10.0 * 4 * A / 1000 * 6.0 * 120.0)}
        v = sc[sc.fixed_valid & sc.fixed_detected]
        if len(v):
            f = v.loc[v.fixed_area_groove_um2.idxmax()]  # fix 5
            row.update(fixed_status="detected", fixed_scan=f.file, fixed_V_mm3=f.fixed_vol_track_mm3,
                       fixed_k=f.fixed_vol_track_mm3 / (r.Load_N * r.S_m))
        else:
            row.update(fixed_status="not_detected" if sc.fixed_valid.any() else "not_measurable", fixed_scan="",
                       fixed_V_mm3=np.nan, fixed_k=np.nan)
        row.update(ours_status=r.wear_status, ours_V_mm3=r.V_mm3, ours_k=r.k_mm3_Nm)
        rows.append(row)
    per_run = pd.DataFrame(rows).merge(cofs, on="run_id").merge(
        runs[["run_id", "COF_whole", "COF_first10s", "COF_instrument_mean", "replicate_group", "role", "Run_time_s", "S_m"]],
        on="run_id")
    per_run["cof_ratio_orig_to_ours"] = per_run.orig_cof_overall_mean / per_run.COF_whole
    return scans, per_run


def bug_evidence(scans: pd.DataFrame, per_run: pd.DataFrame, per_loc: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    ev = []
    r24 = scans[scans.file == "R24.bcrf"].iloc[0]
    ev.append((1, "Orientation chosen from the unlevelled map (tilt decides)",
               f"{int(scans.flipped.sum())} of {len(scans)} scans transposed, so the 'profile' runs along the track. "
               f"R24.bcrf: {'transposed' if r24.flipped else 'not transposed'}, reported depth {r24.orig_depth_max_um:.1f} um "
               f"(a pit), groove area {r24.orig_area_groove_um2:.0f} um^2; with the correct axis the same code gives "
               f"depth {r24.axisfix_depth_max_um:.1f} um, groove area {r24.axisfix_area_groove_um2:.0f} um^2"))
    z = scans[(scans.axisfix_area_groove_um2 == 0) & (scans.axisfix_depth_max_um >= 3)]
    ev.append((2, "Points shallower than the cutoff zeroed one by one",
               f"{len(z)} scans (correct axis) have groove area 0 although the depth is >= 3 um, e.g. "
               + ", ".join(f"{a} ({d:.1f} um)" for a, d in zip(z.file.head(4), z.axisfix_depth_max_um.head(4)))))
    pl = scans.merge(per_loc[["file", "robust__pileup_um2", "robust__valid"]], on="file")
    pl = pl[pl.robust__valid & (pl.robust__pileup_um2 > 0)]
    ratio = (pl.axisfix_area_pileup_um2 / pl.robust__pileup_um2).median()
    ev.append((3, "Pile-up summed over the whole image outside the groove",
               f"median TA pile-up / band pile-up = {ratio:.1f} over {len(pl)} valid scans (texture peaks counted as pile-up)"))
    b = per_loc[per_loc.robust__track_found.astype(bool)].copy()
    ny, e = 600, int(600 * 0.15)
    band = b.robust__band.astype(str).str.strip("[]").str.split(",", expand=True).astype(float)
    worn_shoulder = ((band[0] < e) | (band[1] > ny - e)).sum()
    ev.append((4, "Shoulder (reference) rows never checked to be unworn",
               f"in {int(worn_shoulder)} of {len(b)} scans with a track, the worn band reaches into the outer 15 % rows "
               f"that the TA uses as the unworn baseline"))
    diff = (per_run.orig_scan != per_run.fixed_scan) & (per_run.fixed_status == "detected")
    ev.append((5, "Scan picked by the deepest point (texture valleys)",
               f"the deepest-point scan differs from the largest-groove valid scan in {int(diff.sum())} of "
               f"{int((per_run.fixed_status == 'detected').sum())} runs with a detected groove"))
    r, p = stats.pearsonr(per_run.Freq_Hz, per_run.cof_ratio_orig_to_ours)
    ev.append((6, "COF = mean of the instrument |Fx|/Fz channel; deceleration second not trimmed",
               f"TA COF / offset-free COF = {per_run.cof_ratio_orig_to_ours.mean():.3f} on average "
               f"({100 * (1 - per_run.cof_ratio_orig_to_ours.mean()):.0f} % low); the ratio falls with frequency "
               f"(r = {r:+.2f}, p = {p:.3f})"))
    r04 = per_run.set_index("run_id").loc["R04"]
    ev.append((7, "Hard-coded R01 parameters; friction log = first *.csv; outputs written into the data folder",
               f"R04 with the default arguments (10 N, 6 Hz, 120 s): k = {r04.orig_k_R01_defaults:.2e} instead of "
               f"{r04.orig_k:.2e} mm^3/(N m) ({r04.orig_k_R01_defaults / r04.orig_k:.1f}x); run_pipeline() writes "
               f"all_scans_breakdown.csv and single_scan_summary.csv into each run folder (avoided here)"))
    return pd.DataFrame(ev, columns=["bug", "description", "evidence"])
