"""Batch pipeline: data/raw/<Experiment_ID>/ -> ML-ready tables in data/processed/.

Run-folder convention (304 SS DOE; roles / location / checkpoint from config/file_layout.yaml)
---------------------------------------------------------------------------------------------
data/raw/<Experiment_ID>/
    <RunID>.csv  or  <load>_<freq>.csv   tribometer export, 1000 Hz, both slide steps
    <RunID>[_a|_b|_c].raw                height scans at 4 locations (m, a, b, c) of the
                                         track, each at base / cp1 (t_run/2) / cp2 (t_run)
    companions                           previews (.png), metadata (.txt/.xml/...), SEM
    optional wli_scans.csv               file, cbar_min_um, cbar_max_um (PNG-only scans)

Outputs (exactly these three tables)
------------------------------------
runs_checkpoints.csv   one row per run x checkpoint (the ML table)
cof_cycles.csv         one row per sliding cycle
wear_locations.csv     one row per run x checkpoint x location
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics
from .cof import extract_cof_cycles
from .height_io import read_height
from .layout import classify_run
from .tribometer_csv import TriboRun, read_tribometer_csv
from .wear import measure_track_band, measure_wear_scar, subtract_baseline
from .wli import read_wli_screenshot

RUN_SHEET_COLUMNS = [
    "Experiment_ID", "Session", "Face", "X_mm", "Y_mm", "Load_N", "Freq_Hz", "Run_time_s",
    "Cycles_nominal", "Role", "Replicate_of", "Include",
    "Hardness_HV", "Ra_um", "Ambient_T_C", "RH_pct", "Ball_scar_d1_um", "Ball_scar_d2_um", "Notes",
]
CHECKPOINT_COLUMNS = [
    "Experiment_ID", "Role", "Replicate_of", "Session", "Face", "X_mm", "Y_mm", "Load_N", "Freq_Hz",
    "Run_time_s", "Checkpoint", "Time_s", "Cycles_cum", "Distance_m", "Freq_achieved_Hz",
    "Stroke_amp_mm", "Fz_mean_N", "COF_steady", "COF_std", "Track_width_um", "Track_width_sd_um",
    "n_locations_valid", "Groove_volume_mm3",
    "Groove_volume_se_mm3", "Pileup_volume_mm3", "Net_volume_mm3", "k_mm3_per_Nm",
    "k_net_mm3_per_Nm", "Wear_detected", "Wear_Volume_LOD_mm3", "V_width_mm3", "V_ball_mm3",
    "n_locations", "Hardness_HV", "Ra_um", "Ambient_T_C", "RH_pct", "Params_source", "DOE_row",
    "Test_date", "status", "issues", "flags",
]
CYCLE_COLUMNS = ["Experiment_ID", "Step", "Cycle", "t_s", "COF_c", "Fz_N", "Halfrange_N", "Amp_mm"]
LOCATION_COLUMNS = [
    "Experiment_ID", "Checkpoint", "Time_s", "Location", "file", "baseline_file", "read_method",
    "wear_method", "Wear_detected", "Groove_area_um2", "Pileup_area_um2", "Net_area_um2",
    "Track_width_um", "Track_max_depth_um", "Area_LOD_um2", "Track_angle_deg", "baseline_shift_px",
    "preview_r", "validated", "wear_flags", "Band_width_um", "Groove_unc_um2", "Track_contrast",
    "Sa_unworn_um", "Sa_track_um", "Stage_X_mm", "Stage_Y_mm",
]
SHEET_TO_ROW = ["Role", "Replicate_of", "Session", "Face", "X_mm", "Y_mm", "Load_N", "Freq_Hz",
                "Run_time_s", "Hardness_HV", "Ra_um", "Ambient_T_C", "RH_pct", "Params_source", "DOE_row"]
PARAM_TOL = {"Load_N": 0.02, "Freq_Hz": 0.02, "Run_time_s": 0.03}     # relative, for DOE matching
CP_INDEX = {"cp1": 1, "cp2": 2}


# ---------------------------------------------------------------------------
def _num(row: dict, key: str, default=np.nan) -> float:
    v = row.get(key, default)
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def load_run_sheet(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=RUN_SHEET_COLUMNS)
    rs = pd.read_csv(path, dtype={"Experiment_ID": str, "Face": str, "Replicate_of": str}, comment="#")
    rs["Experiment_ID"] = rs["Experiment_ID"].astype(str).str.strip()
    return rs


def _test_date(s: str) -> str:
    """CSV header TestDate ('10/16/2026 3:11:31 AM') -> ISO '2026-10-16T03:11:31' ('' if absent)."""
    from datetime import datetime
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(s).strip(), fmt).isoformat()
        except ValueError:
            pass
    return str(s).strip()


def included(sheet: pd.DataFrame) -> pd.DataFrame:
    """Run-sheet rows with Include != N."""
    if "Include" not in sheet:
        return sheet
    return sheet[sheet["Include"].astype(str).str.strip().str.upper() != "N"]


def read_cof_csvs(paths: list[Path], cfg: dict) -> TriboRun:
    """One TriboRun from one or more exports (appended in mtime order, steps kept apart)."""
    runs = [read_tribometer_csv(p, cfg["csv"]["columns"]) for p in sorted(paths, key=lambda p: p.stat().st_mtime)]
    if len(runs) == 1:
        return runs[0]
    parts, t_end = [], 0.0
    for i, r in enumerate(runs):
        d = r.data.copy()
        d["recipe_step"] = d["recipe_step"] + 1000 * i
        d["t_s"] = d["t_s"] - d["t_s"].iloc[0] + t_end
        t_end = float(d["t_s"].iloc[-1]) + 1.0 / r.sample_rate_hz
        parts.append(d)
    return TriboRun(path=runs[0].path, meta=runs[0].meta, data=pd.concat(parts, ignore_index=True),
                    column_map=runs[0].column_map, sample_rate_hz=runs[0].sample_rate_hz)


def run_parameters(run_dir: Path, cfg: dict) -> tuple[dict, str, list]:
    """Load_N, Freq_Hz, Run_time_s of one run from its own files.

    Order: the recipe (``.lrx``, what the machine was set to) -> the tribometer CSV (mean Fz
    while sliding, achieved frequency, sliding time). Returns ``(params, source, issues)``.
    """
    from .cof import _reversals, sliding_segments
    from .layout import classify_run
    from .recipe import recipe_params
    by_role, _, _ = classify_run(run_dir, cfg["layout"])
    issues: list[str] = []
    if by_role.get("recipe"):
        try:
            rp = recipe_params(by_role["recipe"][0])
            if all(np.isfinite(rp[k]) for k in ("Load_N", "Freq_Hz", "Run_time_s")):
                if not rp.get("recipe_consistent", True):
                    issues.append("recipe_steps_differ")
                return {k: rp[k] for k in ("Load_N", "Freq_Hz", "Run_time_s")}, "recipe_lrx", issues
        except Exception as exc:                                        # noqa: BLE001
            issues.append(f"recipe_unreadable: {exc}")
    if by_role.get("cof_csv"):
        run = read_cof_csvs(by_role["cof_csv"], cfg)
        d = run.data
        segs = sliding_segments(d)
        if segs:
            idx = np.concatenate(segs)
            dur = float(sum(d["t_s"].to_numpy()[b[-1]] - d["t_s"].to_numpy()[b[0]] + 1 / run.sample_rate_hz
                            for b in segs))
            load = float(np.nanmean(d["fz_N"].to_numpy(float)[idx])) if "fz_N" in d else np.nan
            if "freq_Hz" in d and np.nanmax(np.nan_to_num(d["freq_Hz"].to_numpy(float)[idx])) > 0:
                f = float(np.nanmax(d["freq_Hz"].to_numpy(float)[idx]))
            else:
                sig = d["fx_N" if "fx_N" in d else "cof"].to_numpy(float)
                n_rev = sum(len(_reversals(sig[b], run.sample_rate_hz, 0.02, 0.2)) for b in segs)
                f = n_rev / 2 / max(dur - 1.0 * len(segs), 1e-9)          # 2 x 0.5 s lost in the ramps
            return ({"Load_N": round(load * 2) / 2, "Freq_Hz": round(f * 2) / 2, "Run_time_s": round(dur)},
                    "csv_measured", issues)
    return {}, "none", issues + ["no_recipe_or_csv"]


def match_doe(exp_id: str, params: dict, sheet: pd.DataFrame) -> tuple[dict, str]:
    """Run-sheet row for a run, chosen by its *measured* parameters, not by its name.

    The sheet row with the same ID is used if its (Load_N, Freq_Hz, Run_time_s) agree with
    the files; otherwise the row whose parameters agree (a replicate condition prefers the
    center_replicate row); otherwise none (role 'interior' = training, flag not_in_doe).
    """
    if sheet.empty or not params:
        return {}, "none"

    def agrees(r):
        return all(np.isfinite(_num(r, k)) and abs(_num(r, k) - params[k]) <= tol * params[k]
                   for k, tol in PARAM_TOL.items())
    rows = {r["Experiment_ID"]: r for r in sheet.to_dict("records")}
    if exp_id in rows and agrees(rows[exp_id]):
        return rows[exp_id], exp_id
    hits = [r for r in rows.values() if agrees(r)]
    if not hits:
        return {}, "none"
    hits.sort(key=lambda r: (str(r.get("Role")) != "center_replicate", r["Experiment_ID"]))
    return hits[0], hits[0]["Experiment_ID"]


def _read_scan(scan: pd.Series, cfg: dict, manifest: dict) -> tuple:
    if scan["height"] is not None:
        hm = read_height(scan["height"], cfg)
        return hm, hm.info.get("read_method"), Path(scan["height"]).name
    rec = manifest.get(Path(scan["preview"]).name, {})
    cmin, cmax = _num(rec, "cbar_min_um"), _num(rec, "cbar_max_um")
    hm = read_wli_screenshot(scan["preview"], cfg, cbar_min=cmin if np.isfinite(cmin) else None,
                             cbar_max=cmax if np.isfinite(cmax) else None)
    return hm, "screenshot", Path(scan["preview"]).name


# ---------------------------------------------------------------------------
def measure_locations(exp_id: str, scans: pd.DataFrame, cfg: dict, sheet_row: dict, t_run: float,
                      hertz_w: float, rep_dir: Path, make_plots: bool, issues: list) -> list[dict]:
    """Groove / pile-up per location x checkpoint (baseline-subtracted when the base exists)."""
    manifest = {}
    run_dir = Path(cfg["paths"]["raw_dir"]) / exp_id
    if (run_dir / "wli_scans.csv").exists():
        manifest = {str(r.get("file")).strip(): r for r in pd.read_csv(run_dir / "wli_scans.csv").to_dict("records")}
    ang_cfg = sheet_row.get("track_angle_deg", cfg["wear"]["track_angle_deg"])
    ang = _num({"a": ang_cfg}, "a") if str(ang_cfg).strip().lower() not in ("auto", "", "nan") else None
    rows = []
    for loc, sub in scans[scans["checkpoint"].notna()].groupby("location"):
        maps = {}
        for _, s in sub.iterrows():
            try:
                maps[s["checkpoint"]] = _read_scan(s, cfg, manifest)
                if maps[s["checkpoint"]][0].info.get("validated") is False:
                    issues.append(f"height_validation_failed[{maps[s['checkpoint']][2]}]")
            except Exception as exc:                                    # noqa: BLE001
                issues.append(f"scan_read_error[{loc}/{s['checkpoint']}]: {exc}")
        base = maps.get("base")
        for cp in ("cp1", "cp2"):
            if cp not in maps:
                continue
            hm, read_method, fname = maps[cp]
            try:
                target, extra, method = hm, {}, "single_scan"
                if cfg["wear"].get("method", "groove") == "band":
                    method = "band"
                    res, wint = measure_track_band(hm, cfg)
                else:
                    if base is not None and cfg["wear"].get("use_baseline_difference", True):
                        target, extra = subtract_baseline(hm, base[0])
                        method = "baseline_difference"
                    res, wint = measure_wear_scar(target, cfg, angle_deg=ang, expected_width_um=hertz_w)
            except Exception as exc:                                    # noqa: BLE001
                issues.append(f"wear_error[{loc}/{cp}]: {exc}")
                continue
            det = bool(res["wear_detected"])
            stage = (np.nan, np.nan)
            meta = sub.loc[sub["checkpoint"] == cp, "metadata"]
            if len(meta) and meta.iloc[0] is not None and str(meta.iloc[0]).lower().endswith(".info"):
                try:
                    from .recipe import read_info
                    inf = read_info(meta.iloc[0])
                    stage = (float(inf.get("XPos", "nan")), float(inf.get("YPos", "nan")))
                except Exception:                                       # noqa: BLE001
                    pass
            rows.append({
                "Experiment_ID": exp_id, "Checkpoint": CP_INDEX[cp], "Time_s": t_run * CP_INDEX[cp] / 2,
                "Location": loc, "file": fname, "baseline_file": base[2] if base is not None else None,
                "read_method": read_method, "wear_method": method, "Wear_detected": det,
                "Groove_area_um2": res["worn_area_um2"] if det else np.nan,
                "Pileup_area_um2": res.get("pileup_area_um2", np.nan) if det else np.nan,
                "Net_area_um2": res.get("net_area_um2", np.nan) if det else np.nan,
                "Track_width_um": res.get("track_width_um", np.nan) if det else np.nan,
                "Track_max_depth_um": res.get("track_max_depth_um", np.nan) if det else np.nan,
                "Area_LOD_um2": res["area_LOD_um2"], "Track_angle_deg": res["track_angle_deg"],
                "baseline_shift_px": extra.get("baseline_shift_px"),
                "preview_r": hm.info.get("preview_r", np.nan), "validated": hm.info.get("validated"),
                "wear_flags": res["wear_flags"], "Band_width_um": res.get("band_width_um", np.nan),
                "Groove_unc_um2": res.get("groove_unc_um2", np.nan), "Track_contrast": res.get("contrast", np.nan),
                "Sa_unworn_um": res.get("Sa_unworn_um", np.nan), "Sa_track_um": res.get("Sa_track_um", np.nan),
                "Stage_X_mm": stage[0], "Stage_Y_mm": stage[1]})
            if make_plots:
                from .plots import plot_band, plot_wear
                title = f"{exp_id} — location {loc}, {cp} ({method})"
                if method == "band":
                    plot_band(wint, res, title, rep_dir / f"wear_{loc}_{cp}.png")
                else:
                    plot_wear(target, wint, res, title, rep_dir / f"wear_{loc}_{cp}.png")
    return rows


def aggregate_wear(locs: pd.DataFrame, ell_mm: float, R_mm: float, tol: float) -> tuple[dict, list]:
    """Per-checkpoint groove / pile-up volumes from the location rows (mean area x l)."""
    if len(locs) == 0:                                   # this checkpoint was not scanned
        return {"n_locations": 0, "Wear_detected": np.nan}, ["not_scanned"]
    if (locs["wear_method"] == "band").all():
        return aggregate_band(locs, ell_mm, R_mm)
    flags: list[str] = []
    out = {"n_locations": int(len(locs))}
    det = locs[locs["Wear_detected"].astype(bool)]
    out["Wear_detected"] = bool(len(det)) if len(locs) else np.nan      # no scan -> unknown, not < LOD
    out["Wear_Volume_LOD_mm3"] = float(locs["Area_LOD_um2"].mean() * 1e-6 * ell_mm) if len(locs) else np.nan
    if len(det) == 0:
        out.update({k: np.nan for k in ("Groove_volume_mm3", "Groove_volume_se_mm3", "Pileup_volume_mm3",
                                        "Net_volume_mm3", "V_width_mm3")})
        if len(locs):
            flags.append("below_LOD")
        return out, flags
    if len(det) < len(locs):
        flags.append(f"detected_{len(det)}_of_{len(locs)}_locations")
    g = det["Groove_area_um2"].to_numpy(float)
    out["Groove_volume_mm3"] = float(g.mean() * 1e-6 * ell_mm)
    out["Groove_volume_se_mm3"] = float(g.std(ddof=1) / np.sqrt(len(g)) * 1e-6 * ell_mm) if len(g) > 1 else np.nan
    out["Pileup_volume_mm3"] = float(det["Pileup_area_um2"].mean() * 1e-6 * ell_mm)
    out["Net_volume_mm3"] = out["Groove_volume_mm3"] - out["Pileup_volume_mm3"]
    w = float(det["Track_width_um"].mean()) / 1000.0
    out["V_width_mm3"] = ell_mm * w ** 3 / (12.0 * R_mm)
    if abs(out["V_width_mm3"] / out["Groove_volume_mm3"] - 1) > tol:
        flags.append("width_model_mismatch")
    return out, flags


def aggregate_band(locs: pd.DataFrame, ell_mm: float, R_mm: float) -> tuple[dict, list]:
    """Run-level wear from rough-surface ("band") location rows.

    * track width: mean over every location where a track was found (no reference needed);
    * groove / pile-up: mean over the *valid* locations only (track found and unworn surface
      on both sides of it); a location's measured area is used even when small, so the mean
      is not biased by keeping only the large ones;
    * detection: run groove > 2 x its reference uncertainty (sqrt(mean u^2) / sqrt(n)); below
      that ``Wear_detected = False`` (censored, "< LOD"); no valid location -> unknown (NaN).
    """
    flags: list[str] = []
    found = locs[locs["Wear_detected"].astype(bool)]
    valid = found[~found["wear_flags"].fillna("").str.contains("reference_one_sided")]
    out = {"n_locations": int(len(locs)), "n_locations_valid": int(len(valid))}
    if len(found):
        w = found["Track_width_um"].to_numpy(float)
        out["Track_width_um"] = float(w.mean())
        out["Track_width_sd_um"] = float(w.std(ddof=1)) if len(w) > 1 else np.nan
        out["V_width_mm3"] = ell_mm * (w.mean() / 1000.0) ** 3 / (12.0 * R_mm)
    if len(found) < len(locs):
        flags.append(f"track_found_{len(found)}_of_{len(locs)}_locations")
    nan = {k: np.nan for k in ("Groove_volume_mm3", "Groove_volume_se_mm3", "Pileup_volume_mm3",
                               "Net_volume_mm3", "Wear_Volume_LOD_mm3")}
    if len(valid) == 0:
        out.update(nan)
        out["Wear_detected"] = np.nan                      # not measurable != below LOD
        flags.append("wear_not_measurable_one_sided_reference" if len(found) else "no_track")
        return out, flags
    if len(valid) < len(found):
        flags.append(f"reference_ok_{len(valid)}_of_{len(found)}_locations")
    g = valid["Groove_area_um2"].to_numpy(float)
    p = valid["Pileup_area_um2"].to_numpy(float)
    u = valid["Groove_unc_um2"].to_numpy(float)
    lod = 2.0 * float(np.sqrt(np.nanmean(u ** 2)) / np.sqrt(len(u)))
    det = bool(g.mean() > lod)
    out.update({
        "Wear_detected": det, "Wear_Volume_LOD_mm3": lod * 1e-6 * ell_mm,
        "Groove_volume_mm3": float(g.mean() * 1e-6 * ell_mm) if det else np.nan,
        "Groove_volume_se_mm3": float(g.std(ddof=1) / np.sqrt(len(g)) * 1e-6 * ell_mm) if len(g) > 1 else np.nan,
        "Pileup_volume_mm3": float(p.mean() * 1e-6 * ell_mm),
        "Net_volume_mm3": float((g.mean() - p.mean()) * 1e-6 * ell_mm)})
    if not det:
        flags.append("below_LOD")
    return out, flags


def process_run(run_dir: Path, sheet_row: dict, cfg: dict, make_plots: bool = True):
    """One run folder -> (checkpoint rows, cycles DataFrame | None, location rows, summary)."""
    exp_id = run_dir.name
    rep_dir = Path(cfg["paths"]["reports_dir"]) / exp_id
    rep_dir.mkdir(parents=True, exist_ok=True)
    t_cfg, w_cfg = cfg["test"], cfg["wear"]
    load, freq, t_run = _num(sheet_row, "Load_N"), _num(sheet_row, "Freq_Hz"), _num(sheet_row, "Run_time_s")
    amp_nom = _num(sheet_row, "Stroke_mm", t_cfg["stroke_mm"])
    R_mm = _num(sheet_row, "Ball_diam_mm", t_cfg["ball_diameter_mm"]) / 2
    by_role, scans, issues = classify_run(run_dir, cfg["layout"], t_run)
    if not sheet_row:
        issues.append("not_in_run_sheet")
    run_flags: list[str] = []

    # ---------------- per-cycle COF ------------------------------------------
    cps, cycles, test_date = {}, None, ""
    if not by_role["cof_csv"]:
        issues.append("no_cof_csv")
    else:
        if len(by_role["cof_csv"]) > 1:
            issues.append(f"{len(by_role['cof_csv'])}_cof_csvs_appended")
        run = read_cof_csvs(by_role["cof_csv"], cfg)
        test_date = _test_date(run.meta.get("TestDate", ""))
        cp_list, cycles, cint = extract_cof_cycles(run, cfg, load_N=load, freq_hz=freq, stroke_mm=amp_nom)
        cps = {c["Checkpoint"]: c for c in cp_list}
        run_flags += cint["flags"]
        if make_plots and len(cycles):
            from .plots import plot_cof_cycles
            plot_cof_cycles(cycles, cp_list, exp_id, rep_dir / "cof_cycles.png")

    # ---------------- wear per location --------------------------------------
    hz = physics.hertz_contact(load, 2 * R_mm, t_cfg["E_ball_GPa"], t_cfg["nu_ball"],
                               t_cfg["E_flat_GPa"], t_cfg["nu_flat"])
    if scans.empty or not scans["checkpoint"].notna().any():
        issues.append("no_height_scans")
    loc_rows = measure_locations(exp_id, scans, cfg, sheet_row, t_run, hz["hertz_width_um"], rep_dir,
                                 make_plots, issues) if len(scans) else []
    locs = pd.DataFrame(loc_rows, columns=LOCATION_COLUMNS)

    # ---------------- one row per checkpoint ---------------------------------
    rows = []
    for j in (1, 2):
        row = {"Experiment_ID": exp_id, **{k: sheet_row.get(k, np.nan) for k in SHEET_TO_ROW},
               "Checkpoint": j, "Time_s": t_run * j / 2, "Test_date": test_date}
        flags = list(run_flags)
        c = cps.get(j)
        if c is not None and c["Cycles_step"] > 0:
            row.update({k: c[k] for k in ("Cycles_cum", "Distance_m", "Freq_achieved_Hz", "Stroke_amp_mm",
                                          "Fz_mean_N", "COF_steady", "COF_std")})
        else:
            n_nom = freq * t_run * j / 2
            row.update({"Cycles_cum": n_nom, "Distance_m": 4 * amp_nom * n_nom / 1000.0,
                        "Stroke_amp_mm": amp_nom})
            flags.append("distance_nominal")
        w, wflags = aggregate_wear(locs[locs["Checkpoint"] == j], w_cfg["track_length_mm"], R_mm,
                                   w_cfg["width_check_tol"])
        row.update(w)
        flags += wflags
        S = row["Distance_m"]
        row["k_mm3_per_Nm"] = physics.specific_wear_rate(row.get("Groove_volume_mm3", np.nan), load, S)
        row["k_net_mm3_per_Nm"] = physics.specific_wear_rate(row.get("Net_volume_mm3", np.nan), load, S)
        d1, d2 = _num(sheet_row, "Ball_scar_d1_um"), _num(sheet_row, "Ball_scar_d2_um")
        d_mm = (d1 + d2) / 2 / 1000.0
        row["V_ball_mm3"] = np.pi * d_mm ** 4 / (64 * R_mm) if j == 2 and np.isfinite(d_mm) else np.nan
        ok = np.isfinite(_num(row, "COF_steady")) and np.isfinite(_num(row, "k_mm3_per_Nm"))
        row["status"] = "ok" if ok else "partial"
        row["issues"] = ";".join(issues)
        row["flags"] = ";".join(dict.fromkeys(flags))
        rows.append(row)

    summary = {"Experiment_ID": exp_id, "issues": issues, "files": {k: [p.name for p in v] for k, v in by_role.items() if v},
               "scans": scans.drop(columns=["dir"]).astype(str).to_dict("records") if len(scans) else [],
               "cof_steps": list(cps.values()), "checkpoints": rows}
    with open(rep_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda v: None if isinstance(v, float) and not np.isfinite(v) else str(v))
    if cycles is not None:
        cycles = cycles.copy()
        cycles.insert(0, "Experiment_ID", exp_id)
    return rows, cycles, loc_rows, summary


def _merge(new: pd.DataFrame, path: Path, ids: set, only) -> pd.DataFrame:
    if only and path.exists():
        old = pd.read_csv(path, dtype={"Experiment_ID": str})
        new = pd.concat([old[~old["Experiment_ID"].isin(ids)], new], ignore_index=True)
    return new


def process_all(cfg: dict, only: list[str] | None = None, make_plots: bool = True, log=print):
    """Every run folder in ``paths.raw_dir`` whose name starts with a run ID.

    Load, frequency and run time come from the run's own files (recipe .lrx, else the CSV);
    the run sheet is only used for what the files cannot tell (role, plate position, manual
    measurements), matched by those parameters (:func:`match_doe`). Replicates are the runs
    whose parameters are identical. A run missing from the folder is simply absent.
    """
    import re
    raw = Path(cfg["paths"]["raw_dir"])
    full_sheet = load_run_sheet(cfg["paths"]["run_sheet"])
    excluded = set(full_sheet["Experiment_ID"]) - set(included(full_sheet)["Experiment_ID"])
    rid = re.compile(cfg["layout"]["run_id_regex"] + r"$", re.I)
    run_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and rid.match(d.name)) if raw.exists() else []
    if only:
        run_dirs = [d for d in run_dirs if d.name in only]
    cp_rows, cycles, loc_rows, done, params_of = [], [], [], set(), {}
    for d in run_dirs:
        if d.name in excluded:
            log(f"[   skip] {d.name}: Include = N in the run sheet")
            continue
        if not any(p.is_file() for p in d.rglob("*")):
            continue                               # empty placeholder folder
        done.add(d.name)
        try:
            params, src, p_issues = run_parameters(d, cfg)
            doe, doe_id = match_doe(d.name, params, included(full_sheet))
            row = {k: v for k, v in doe.items() if not (isinstance(v, float) and np.isnan(v))}
            if doe_id not in (d.name, "none"):
                p_issues.append(f"files_match_doe_row_{doe_id}")
            if doe_id == "none":
                p_issues.append("not_in_doe")
                row = {"Role": "interior"}
            if doe_id != d.name:                   # position / manual columns belong to another run
                for k in ("Session", "Face", "X_mm", "Y_mm", "Hardness_HV", "Ra_um", "Ambient_T_C",
                          "RH_pct", "Ball_scar_d1_um", "Ball_scar_d2_um"):
                    row.pop(k, None)
                own = full_sheet[full_sheet["Experiment_ID"] == d.name]
                for k in ("Hardness_HV", "Ra_um", "Ambient_T_C", "RH_pct", "Ball_scar_d1_um", "Ball_scar_d2_um"):
                    if len(own) and pd.notna(own.iloc[0].get(k)):
                        row[k] = own.iloc[0][k]
            row.update(params)
            row["Params_source"], row["DOE_row"] = src, doe_id
            row.pop("Replicate_of", None)          # set below from identical parameters
            params_of[d.name] = tuple(params.get(k) for k in ("Load_N", "Freq_Hz", "Run_time_s"))
            rows, cyc, locs, _ = process_run(d, row, cfg, make_plots)
            for r in rows:
                if p_issues:
                    r["issues"] = ";".join([*p_issues, *(r["issues"].split(";") if r["issues"] else [])])
                log(f"[{r['status']:>7}] {d.name} cp{r['Checkpoint']}  L={r['Load_N']} f={r['Freq_Hz']} "
                    f"t={r['Run_time_s']} ({src})  COF={_num(r, 'COF_steady'):.4f}  "
                    f"k={_num(r, 'k_mm3_per_Nm'):.3e}  {r['issues']}")
            cp_rows += rows
            loc_rows += locs
            if cyc is not None:
                cycles.append(cyc)
        except Exception as exc:                   # noqa: BLE001
            for j in (1, 2):
                cp_rows.append({"Experiment_ID": d.name, "Checkpoint": j, "status": "error", "issues": repr(exc)})
            log(f"[  error] {d.name}: {exc}")
            log(traceback.format_exc(limit=3))

    # replicates = identical (Load_N, Freq_Hz, Run_time_s) read from the files
    by_cond: dict = {}
    for rid_, cond in sorted(params_of.items()):
        if None not in cond and all(np.isfinite(cond)):
            by_cond.setdefault(cond, []).append(rid_)
    rep_of = {r: ids[0] for ids in by_cond.values() if len(ids) > 1 for r in ids[1:]}
    for r in cp_rows:
        r["Replicate_of"] = rep_of.get(r["Experiment_ID"], np.nan)
    reps = [ids for ids in by_cond.values() if len(ids) > 1]
    if reps:
        log("replicate conditions (identical L, f, t in the files): " + "; ".join("/".join(i) for i in reps))

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs_checkpoints.csv": (pd.DataFrame(cp_rows, columns=CHECKPOINT_COLUMNS), ["Experiment_ID", "Checkpoint"]),
        "cof_cycles.csv": (pd.concat(cycles, ignore_index=True)[CYCLE_COLUMNS] if cycles
                           else pd.DataFrame(columns=CYCLE_COLUMNS), ["Experiment_ID", "Step", "Cycle"]),
        "wear_locations.csv": (pd.DataFrame(loc_rows, columns=LOCATION_COLUMNS),
                               ["Experiment_ID", "Checkpoint", "Location"]),
    }
    res = {}
    for name, (df, key) in tables.items():
        df = _merge(df, out / name, done, only)
        df = df.sort_values(key).reset_index(drop=True) if len(df) else df
        df.to_csv(out / name, index=False)
        res[name] = df
    log(f"\nwrote {out}/runs_checkpoints.csv ({len(res['runs_checkpoints.csv'])} rows), cof_cycles.csv "
        f"({len(res['cof_cycles.csv'])}), wear_locations.csv ({len(res['wear_locations.csv'])})")
    return res["runs_checkpoints.csv"]
