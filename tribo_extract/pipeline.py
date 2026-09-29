"""Batch pipeline: data/raw/<Experiment_ID>/ -> ML-ready tables in data/processed/.

Run-folder convention
---------------------
data/raw/<Experiment_ID>/
    <anything>.csv                 tribometer export (exactly one)
    <anything>.png                 WLI screenshot of the wear track after the test
    optional extra scans           name them with the elapsed sliding time, e.g.
                                   scan_t0s.png, scan_t60s.png ... scan_t600s.png
                                   ("baseline" / "pre" / t0 -> unworn reference)
    optional raw height export     <name>_height.csv / .txt / .asc (preferred over png)
    optional wli_scans.csv         columns: file, t_s, cbar_min_um, cbar_max_um
                                   (per-scan time / colour-bar limits when OCR is unavailable)
"""
from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics
from .cof import extract_cof
from .tribometer_csv import read_tribometer_csv
from .wear import level_plane, measure_wear_scar, subtract_baseline
from .wli import read_height_matrix, read_wli_screenshot

RUN_SHEET_COLUMNS = [
    "Experiment_ID", "Carbon_pct", "Hardness_HV", "Load_N", "Freq_Hz", "Stroke_mm", "Duration_s",
    "Ball_diam_mm", "Ball_ID", "Specimen_ID", "Replicate_of", "Ambient_T_C", "RH_pct",
    "Initial_Ra_um", "cbar_min_um", "cbar_max_um", "fov_x_um", "fov_y_um", "track_angle_deg",
    "sliding_step", "Notes",
]
# column order of runs_targets.csv: id -> inputs/features -> targets -> details/QC
LEAD_COLUMNS = [
    "Experiment_ID", "status",
    "Carbon_pct", "Hardness_HV", "Load_N", "Freq_Hz", "Stroke_mm", "Speed_mps", "sliding_time_s",
    "N_cycles", "Sliding_distance_m", "Hertz_width_um", "Hertz_p_mean_MPa", "Ambient_T_C", "RH_pct",
    "Initial_Ra_um", "Initial_Sa_um",
    "COF_ss_mean", "COF_ss_median", "COF_ss_std", "t_runin_s", "COF_runin_peak",
    "Specific_Wear_Rate_mm3_per_Nm", "Wear_Volume_mm3", "Wear_detected", "Wear_method",
    "Wear_Volume_LOD_mm3", "Specific_Wear_Rate_LOD_mm3_per_Nm",
    "Specific_Wear_Rate_slope_mm3_per_Nm", "Vdot_steady_mm3_per_s", "dV_run_mm3", "tau_trans_s",
]
TRAIL_COLUMNS = ["issues", "cof_flags", "wear_flags", "Notes"]

FEATURE_COLUMNS = ["Carbon_pct", "Hardness_HV", "Load_N", "Freq_Hz", "Stroke_mm", "Speed_mps",
                   "Hertz_p_mean_MPa", "Ambient_T_C", "RH_pct", "Initial_Ra_um"]


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
    rs = pd.read_csv(path, dtype={"Experiment_ID": str}, comment="#")
    rs["Experiment_ID"] = rs["Experiment_ID"].astype(str).str.strip()
    return rs


def _scan_time(name: str):
    n = name.lower()
    if re.search(r"baseline|(^|[_-])pre([_.-]|$)", n):
        return 0.0
    m = re.search(r"(?:^|[_-])t(\d+(?:\.\d+)?)s?(?=[_.-]|$)", n)
    return float(m.group(1)) if m else None


def discover_run(run_dir: Path) -> dict:
    csvs = sorted(p for p in run_dir.glob("*.csv")
                  if not re.search(r"height", p.name, re.I) and p.name != "wli_scans.csv")
    heights = sorted(p for p in run_dir.iterdir()
                     if re.search(r"height", p.name, re.I) and p.suffix.lower() in (".csv", ".txt", ".asc"))
    pngs = sorted(list(run_dir.glob("*.png")) + list(run_dir.glob("*.PNG")))
    return {"csv": csvs, "png": pngs, "height": heights}


# ---------------------------------------------------------------------------
def process_run(run_dir: Path, sheet_row: dict, cfg: dict, make_plots: bool = True):
    """Process one run folder. Returns (targets_row: dict, cof_series: DataFrame|None)."""
    exp_id = run_dir.name
    rep_dir = Path(cfg["paths"]["reports_dir"]) / exp_id
    rep_dir.mkdir(parents=True, exist_ok=True)
    files = discover_run(run_dir)
    row: dict = {"Experiment_ID": exp_id}
    for k in RUN_SHEET_COLUMNS[1:]:
        if k in sheet_row:
            row[k] = sheet_row[k]
    issues: list[str] = []
    if not sheet_row:
        issues.append("not_in_run_sheet")

    t_cfg = cfg["test"]
    load = _num(sheet_row, "Load_N")
    freq = _num(sheet_row, "Freq_Hz")
    stroke = _num(sheet_row, "Stroke_mm", t_cfg["stroke_mm"])
    ball_d = _num(sheet_row, "Ball_diam_mm", t_cfg["ball_diameter_mm"])
    sdef = t_cfg["stroke_definition"]
    row["Stroke_mm"] = stroke
    cfg_run = json.loads(json.dumps(cfg, default=str))          # per-run copy
    if str(sheet_row.get("sliding_step", "")).strip() not in ("", "nan"):
        cfg_run["csv"]["sliding_step"] = sheet_row["sliding_step"]

    # ---------------- COF from CSV ------------------------------------------
    ts = None
    sliding_time = _num(sheet_row, "Duration_s")
    if len(files["csv"]) != 1:
        issues.append(f"expected 1 csv, found {len(files['csv'])}")
    if files["csv"]:
        run = read_tribometer_csv(files["csv"][0], cfg["csv"]["columns"])
        cof_res, ts, cint = extract_cof(run, cfg_run, load_N=load)
        row.update(cof_res)
        row["csv_file"] = files["csv"][0].name
        row["test_date"] = run.meta.get("TestDate")
        row["recipe"] = run.meta.get("Recipe")
        sliding_time = cof_res["sliding_time_s"]
        dur_plan = _num(sheet_row, "Duration_s")
        if np.isfinite(dur_plan) and abs(sliding_time - dur_plan) / dur_plan > 0.02:
            issues.append("duration_differs_from_plan")
        if not np.isfinite(load):
            load = cof_res.get("Fz_mean_N", np.nan)
            row["Load_N"] = round(load, 3)
            issues.append("Load_N_from_measured_Fz")
        if make_plots:
            from .plots import plot_cof
            plot_cof(cint, cof_res, exp_id, rep_dir / "cof.png")
            ts.to_csv(rep_dir / "cof_timeseries.csv", index=False)

    # ---------------- derived kinematics / contact --------------------------
    if np.isfinite(freq):
        row["Speed_mps"] = physics.mean_speed_mps(freq, stroke, sdef)
        if np.isfinite(sliding_time):
            row["N_cycles"] = freq * sliding_time
            row["Sliding_distance_m"] = physics.sliding_distance_m(freq, stroke, sliding_time, sdef)
    else:
        issues.append("missing_Freq_Hz")
    hz = physics.hertz_contact(load, ball_d, t_cfg["E_ball_GPa"], t_cfg["nu_ball"],
                               t_cfg["E_flat_GPa"], t_cfg["nu_flat"])
    row["Hertz_width_um"] = hz["hertz_width_um"]
    row["Hertz_p_mean_MPa"] = hz["p_mean_MPa"]
    track_len_mm = physics.distance_per_cycle_mm(stroke, sdef) / 2

    # ---------------- wear from WLI ------------------------------------------
    scans = []
    for p in files["height"] + files["png"]:
        scans.append({"path": p, "t": _scan_time(p.stem), "kind": "height" if p in files["height"] else "png"})
    # a raw height export supersedes the screenshot of the same checkpoint
    if files["height"]:
        ht = {s["t"] for s in scans if s["kind"] == "height"}
        scans = [s for s in scans if s["kind"] == "height" or s["t"] not in ht]
    # optional per-scan table in the run folder: file, t_s, cbar_min_um, cbar_max_um
    manifest = {}
    if (run_dir / "wli_scans.csv").exists():
        for rec in pd.read_csv(run_dir / "wli_scans.csv").to_dict("records"):
            manifest[str(rec.get("file")).strip()] = rec
    for s in scans:
        rec = manifest.get(s["path"].name, {})
        if np.isfinite(_num(rec, "t_s")):
            s["t"] = _num(rec, "t_s")
        s["cbar"] = (_num(rec, "cbar_min_um"), _num(rec, "cbar_max_um"))
        if s["t"] is None:
            s["t"] = sliding_time              # untagged scan = end of test
    if not scans:
        issues.append("no_wli_scan")
    single = len(scans) == 1
    ang_cfg = sheet_row.get("track_angle_deg", cfg["wear"]["track_angle_deg"])
    ang = _num({"a": ang_cfg}, "a") if str(ang_cfg).strip().lower() not in ("auto", "", "nan") else None
    # load every scan first (the baseline is needed before the others)
    maps = []
    for s in sorted(scans, key=lambda s: s["t"]):
        try:
            if s["kind"] == "png":
                fov = None
                if np.isfinite(_num(sheet_row, "fov_x_um")) and np.isfinite(_num(sheet_row, "fov_y_um")):
                    fov = [_num(sheet_row, "fov_x_um"), _num(sheet_row, "fov_y_um")]
                is_final = s["t"] == max(x["t"] for x in scans)
                cmin, cmax = s["cbar"]
                if not np.isfinite(cmin) and (single or is_final):     # run sheet = final scan
                    cmin, cmax = _num(sheet_row, "cbar_min_um"), _num(sheet_row, "cbar_max_um")
                hm = read_wli_screenshot(
                    s["path"], cfg,
                    cbar_min=cmin if np.isfinite(cmin) else None,
                    cbar_max=cmax if np.isfinite(cmax) else None,
                    fov_um=fov)
            else:
                fx, fy = cfg["wli"]["fov_um"][cfg["wli"]["default_objective"]]
                hm = read_height_matrix(s["path"], [_num(sheet_row, "fov_x_um", fx),
                                                    _num(sheet_row, "fov_y_um", fy)])
            maps.append((s, hm))
        except Exception as exc:                                    # noqa: BLE001
            issues.append(f"wli_error[{s['path'].name}]: {exc}")
    base_hm = next((hm for s, hm in maps if s["t"] == 0), None) if len(maps) > 1 else None
    use_diff = base_hm is not None and cfg["wear"].get("use_baseline_difference", True)

    wear_rows = []
    for s, hm in maps:
        try:
            method = "single_scan"
            target, extra = hm, {}
            if use_diff and hm is not base_hm:
                target, extra = subtract_baseline(hm, base_hm)
                method = "baseline_difference"
            wres, wint = measure_wear_scar(target, cfg, angle_deg=ang, expected_width_um=hz["hertz_width_um"])
            zl = level_plane(hm.z, 1)                     # areal roughness of the scan itself
            Sa = float(np.nanmean(np.abs(zl - np.nanmean(zl))))
            Sq = float(np.sqrt(np.nanmean((zl - np.nanmean(zl)) ** 2)))
            area = wres["net_area_um2"] if (cfg["wear"]["subtract_pileup"] and wres["wear_detected"]) \
                else wres["worn_area_um2"]
            vol = area * 1e-6 * track_len_mm if wres["wear_detected"] else np.nan
            wear_rows.append({"scan": s["path"].name, "t_s": s["t"], "wear_method": method, **extra,
                              "volume_mm3": vol,
                              "volume_LOD_mm3": wres["area_LOD_um2"] * 1e-6 * track_len_mm,
                              "Sa_um": Sa, "Sq_um": Sq, "cbar": [hm.zmin, hm.zmax],
                              "scan_id": hm.info.get("scan_id"), "masked_frac": hm.masked_frac,
                              "cbar_warning": hm.info.get("warn_cbar"), **wres})
            if make_plots:
                from .plots import plot_wear
                plot_wear(target, wint, wres, f"{exp_id} — {s['path'].name} (t = {s['t']} s, {method})",
                          rep_dir / f"wear_{s['path'].stem}.png")
        except Exception as exc:                                    # noqa: BLE001
            issues.append(f"wear_error[{s['path'].name}]: {exc}")

    if wear_rows:
        final = max(wear_rows, key=lambda r: r["t_s"])
        base = [r for r in wear_rows if r["t_s"] == 0]
        row["Initial_Sa_um"] = base[0]["Sa_um"] if base else np.nan
        if not np.isfinite(_num(sheet_row, "Initial_Ra_um")) and base:
            row["Initial_Ra_um"] = base[0]["Sa_um"]
        row.update({
            "wli_scan": final["scan"], "wli_scan_id": final.get("scan_id"),
            "cbar_min_um": final["cbar"][0], "cbar_max_um": final["cbar"][1],
            "Track_angle_deg": final["track_angle_deg"],
            "Track_width_um": final.get("track_width_um", np.nan),
            "Track_max_depth_um": final.get("track_max_depth_um", np.nan),
            "Worn_area_um2": final.get("worn_area_um2", np.nan),
            "Pileup_area_um2": final.get("pileup_area_um2", np.nan),
            "Wear_method": final["wear_method"],
            "Wear_detected": final["wear_detected"],
            "Wear_Volume_mm3": final["volume_mm3"],
            "Wear_Volume_LOD_mm3": final["volume_LOD_mm3"],
            "Final_Sa_um": final["Sa_um"],
            "wear_flags": final["wear_flags"],
        })
        if final.get("cbar_warning"):
            issues.append(final["cbar_warning"])
        S = row.get("Sliding_distance_m", np.nan)
        row["Specific_Wear_Rate_mm3_per_Nm"] = physics.specific_wear_rate(final["volume_mm3"], load, S)
        row["Specific_Wear_Rate_LOD_mm3_per_Nm"] = physics.specific_wear_rate(final["volume_LOD_mm3"], load, S)
        # time-resolved trajectory when checkpoint scans exist
        if len(wear_rows) >= 2:
            tv = [(r["t_s"], 0.0 if (r["t_s"] == 0 and not r["wear_detected"]) else r["volume_mm3"])
                  for r in wear_rows]
            traj = physics.fit_wear_trajectory([a for a, _ in tv], [b for _, b in tv])
            row.update(traj)
            sp = row.get("Speed_mps", np.nan)
            if "Vdot_linear_mm3_per_s" in traj and np.isfinite(sp):
                row["Specific_Wear_Rate_slope_mm3_per_Nm"] = traj["Vdot_linear_mm3_per_s"] / (load * sp)
        pd.DataFrame([{k: v for k, v in r.items() if k != "cbar"} for r in wear_rows]) \
            .to_csv(rep_dir / "wear_scans.csv", index=False)

    row["issues"] = ";".join(issues)
    row["status"] = ("ok" if (row.get("COF_ss_mean") is not None and np.isfinite(
        row.get("Specific_Wear_Rate_mm3_per_Nm", np.nan))) else "partial")
    with open(rep_dir / "summary.json", "w") as fh:
        json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in row.items()}, fh, indent=2, default=str)

    if ts is not None:
        ts = ts.copy()
        ts.insert(0, "Experiment_ID", exp_id)
        for f in FEATURE_COLUMNS:
            if f in row:
                ts[f] = row[f]
    return row, ts


def process_all(cfg: dict, only: list[str] | None = None, make_plots: bool = True, log=print):
    raw = Path(cfg["paths"]["raw_dir"])
    sheet = load_run_sheet(cfg["paths"]["run_sheet"])
    sheet_map = {r["Experiment_ID"]: {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}
                 for r in sheet.to_dict("records")}
    run_dirs = sorted(d for d in raw.iterdir() if d.is_dir() and not d.name.startswith((".", "_")))
    if only:
        run_dirs = [d for d in run_dirs if d.name in only]
    rows, series = [], []
    for d in run_dirs:
        if not any(d.iterdir()):
            continue                               # empty placeholder folder
        try:
            r, ts = process_run(d, sheet_map.get(d.name, {}), cfg, make_plots)
            log(f"[{r['status']:>7}] {d.name}  COF_ss={r.get('COF_ss_mean', float('nan')):.4f}  "
                f"k={r.get('Specific_Wear_Rate_mm3_per_Nm', float('nan')):.3e}  {r['issues']}")
        except Exception as exc:                   # noqa: BLE001
            r, ts = {"Experiment_ID": d.name, "status": "error", "issues": repr(exc)}, None
            log(f"[  error] {d.name}: {exc}")
            log(traceback.format_exc(limit=3))
        rows.append(r)
        if ts is not None:
            series.append(ts)

    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    if only and (out / "runs_targets.csv").exists():      # merge into the existing table
        old = pd.read_csv(out / "runs_targets.csv", dtype={"Experiment_ID": str})
        runs = pd.concat([old[~old["Experiment_ID"].isin(runs["Experiment_ID"])], runs])
    runs = runs.sort_values("Experiment_ID")
    lead = [c for c in LEAD_COLUMNS if c in runs.columns]
    trail = [c for c in TRAIL_COLUMNS if c in runs.columns]
    runs = runs[lead + [c for c in runs.columns if c not in lead + trail] + trail]
    runs.to_csv(out / "runs_targets.csv", index=False)
    if series:
        cts = pd.concat(series, ignore_index=True)
        if only and (out / "cof_timeseries.csv").exists():
            old = pd.read_csv(out / "cof_timeseries.csv", dtype={"Experiment_ID": str})
            cts = pd.concat([old[~old["Experiment_ID"].isin(cts["Experiment_ID"])], cts])
        cts.sort_values(["Experiment_ID", "Time_s"]).to_csv(out / "cof_timeseries.csv", index=False)
    log(f"\nwrote {out / 'runs_targets.csv'} ({len(runs)} runs)"
        + (f" and {out / 'cof_timeseries.csv'}" if series else ""))
    return runs
