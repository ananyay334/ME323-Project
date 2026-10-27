"""Extraction of every run (cached, parallel) and assembly of the one-row-per-run dataset."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import Paths
from ..io.bcrf import read_bcrf
from ..io.csv_reader import read_friction_csv
from ..io.info import read_info
from ..io.layout import RunFiles, find_runs, load_doe, match_doe, replicate_groups
from ..io.lrx import parse_lrx
from ..utils import LOG, quiet_warnings
from . import cache
from .cof import extract_cof, measure_params
from .wear_band import analyse_scan

UM2_TO_MM3_PER_MM = 1e-6  # um^2 -> mm^2

# wear variants measured on every scan (C2): name -> (robust levelling?, high-pass mode)
WEAR_VARIANTS = {"robust": (True, None), "plain": (False, None), "box": (True, "box")}


# --------------------------------------------------------------------------------------------- per run
def _wear_cfg(cfg: dict, hp_mode: str | None) -> dict:
    w = copy.deepcopy(cfg["wear"])
    if hp_mode:
        w["highpass_mode"] = hp_mode
    return w


def extract_run(run: RunFiles, cfg: dict) -> dict:
    quiet_warnings()
    t0 = time.time()
    out: dict = {"folder_id": run.folder_id, "folder": str(run.folder), "notes": []}
    log = read_friction_csv(run.csv)
    if run.lrx is not None:
        params = parse_lrx(run.lrx)
    else:
        params = measure_params(log, cfg["cof"])
        out["notes"].append("parameters measured from CSV (no .lrx)")
    out["params"] = {k: params[k] for k in ("Load_N", "Freq_Hz", "Run_time_s", "source")}
    out["params"]["static_s"] = params.get("static_s", np.nan)
    out["test_date"] = log.test_date.isoformat() if log.test_date else ""
    out["has_freq_channel"] = log.has_freq
    cof_res, cycles = extract_cof(log, cfg["cof"], cfg["experiment"]["stroke_amplitude_mm"], params)
    out["cof"] = cof_res
    out["cycles"] = cycles
    scans = []
    for s in run.scans:
        hm = read_bcrf(s.bcrf)
        info = read_info(s.info) if s.info else {}
        rec = {"scan": s.stem, "location": s.location, "file": s.bcrf.name, "pixel_um": hm.pixel_um,
               "stage_x_mm": info.get("XPos"), "stage_y_mm": info.get("YPos"), "lens": info.get("Lens"),
               "mode": info.get("Mode"), "void_frac": float(np.mean(~np.isfinite(hm.z)))}
        for name, (robust, hp) in WEAR_VARIANTS.items():
            r = analyse_scan(hm.z, hm.pixel_um, _wear_cfg(cfg, hp), robust=robust)
            for k, v in r.scalars().items():
                rec[f"{name}__{k}"] = v
        scans.append(rec)
    out["scans"] = scans
    out["seconds"] = time.time() - t0
    return out


def extract_all(cfg: dict, paths: Paths, force: bool = False) -> list[dict]:
    runs = find_runs(paths.testdata)
    LOG.info("found %d run folders in %s", len(runs), paths.testdata)
    cdir = paths.cache / "extract"
    cfg_parts = {k: cfg[k] for k in ("cof", "wear", "experiment")}
    todo, results = [], {}
    for r in runs:
        key = cache.cache_key([r.csv, r.lrx] + [s.bcrf for s in r.scans], cfg_parts)
        hit = None if force else cache.load(cdir, r.folder_id, key)
        if hit is not None:
            results[r.folder_id] = hit
        else:
            todo.append((r, key))
    LOG.info("extraction: %d cached, %d to compute", len(results), len(todo))
    if todo:
        new = Parallel(n_jobs=cfg.get("n_jobs", -1))(delayed(extract_run)(r, cfg) for r, _ in todo)
        for (r, key), res in zip(todo, new):
            cache.store(cdir, r.folder_id, key, res)
            results[r.folder_id] = res
            LOG.info("  %s extracted in %.1f s", r.folder_id, res["seconds"])
    return [results[r.folder_id] for r in runs]


# --------------------------------------------------------------------------------------------- wear per run
def aggregate_wear(loc: pd.DataFrame, variant: str, load_N: float, S_m: float, track_mm: float,
                   area_col: str = "groove_um2") -> dict:
    """Run-level wear from its scan locations (one wear variant)."""
    p = f"{variant}__"
    valid = loc[p + "valid"].astype(bool)
    found = loc[p + "track_found"].astype(bool)
    area = loc[p + area_col].astype(float)
    lod = loc[p + "lod_um2"].astype(float)
    det = valid & (area > lod) if area_col == "groove_um2" else valid & (area > 0)
    to_mm3 = UM2_TO_MM3_PER_MM * track_mm
    LS = load_N * S_m
    out = {"n_scans": len(loc), "n_valid": int(valid.sum()), "n_detected": int(det.sum()),
           "n_track_found": int(found.sum())}
    if det.any():
        i = area[det].idxmax()
        out.update(status="detected", location=loc.loc[i, "location"], area_um2=area[i], lod_um2=lod[i])
        out["V_mm3"] = area[i] * to_mm3
        out["k"] = out["V_mm3"] / LS
        out["k_upper"] = np.nan
    elif valid.any():
        out.update(status="below_LOD", location="", area_um2=np.nan, lod_um2=lod[valid].max(), V_mm3=np.nan, k=np.nan)
        out["k_upper"] = lod[valid].max() * to_mm3 / LS
    else:
        out.update(status="not_measurable", location="", area_um2=np.nan, lod_um2=np.nan, V_mm3=np.nan, k=np.nan,
                   k_upper=np.nan)
    out["V_upper_mm3"] = out["k_upper"] * LS if np.isfinite(out["k_upper"]) else np.nan
    if valid.any():
        out["V_mean_loc_mm3"] = area[valid].mean() * to_mm3
        out["k_mean_loc"] = out["V_mean_loc_mm3"] / LS if out["V_mean_loc_mm3"] > 0 else np.nan
        g = loc.loc[valid, p + "groove_um2"].sum()
        out["plasticity_ratio"] = loc.loc[valid, p + "pileup_um2"].sum() / g if g > 0 else np.nan
        out["depth_max_um"] = loc.loc[valid, p + "depth_max_um"].max()
        out["Sa_unworn_um"] = loc.loc[valid, p + "sa_unworn_um"].mean()
    else:
        out.update(V_mean_loc_mm3=np.nan, k_mean_loc=np.nan, plasticity_ratio=np.nan, depth_max_um=np.nan,
                   Sa_unworn_um=loc[p + "sa_unworn_um"].astype(float).mean())
    w = loc.loc[found, p + "width_um"].astype(float)
    out["width_um"] = w.mean() if len(w) else np.nan
    out["width_sd_um"] = w.std(ddof=1) if len(w) > 1 else np.nan
    return out


# --------------------------------------------------------------------------------------------- dataset
def build_dataset(extracted: list[dict], cfg: dict, paths: Paths) -> dict[str, pd.DataFrame]:
    doe = load_doe(paths.inputs / "run_sheet.csv")
    A_mm = cfg["experiment"]["stroke_amplitude_mm"]
    track_mm = cfg["experiment"]["track_length_mm"]
    taken: set[str] = set()
    rows, cyc_all, loc_all, var_rows = [], [], [], []
    # match in folder order; identical (replicate) rows are resolved by folder name first
    order = sorted(extracted, key=lambda e: (match_doe(e["params"], doe, e["folder_id"], cfg["doe_match"]) != e["folder_id"], e["folder_id"]))
    for e in order:
        p = e["params"]
        rid = match_doe(p, doe, e["folder_id"], cfg["doe_match"], taken)
        flags = []
        if rid is None:
            rid = e["folder_id"]
            flags.append("no_doe_match")
        elif rid != e["folder_id"]:
            flags.append(f"folder_{e['folder_id']}_holds_{rid}")
        taken.add(rid)
        d = doe[doe.Experiment_ID == rid]
        drow = d.iloc[0] if len(d) else None
        c = e["cof"]
        S = c["S_m"]
        row = {
            "run_id": rid,
            "folder_id": e["folder_id"],
            "role": "test" if drow is not None and str(drow.Role) == "test" else "training",
            "doe_role": str(drow.Role) if drow is not None else "",
            "test_date": e["test_date"][:10],
            "Load_N": p["Load_N"],
            "Freq_Hz": p["Freq_Hz"],
            "Run_time_s": p["Run_time_s"],
            "params_source": p["source"],
            "static_s": p.get("static_s", np.nan),
            "cycles": c["cycles"],
            "cycles_ratio_ft": c.get("cycles_ratio_ft", np.nan),
            "S_m": S,
            "v_mean_m_s": 4 * A_mm / 1000 * p["Freq_Hz"],
        }
        for k in ("COF_first10s", "COF_0.5_2m", "COF_end", "COF_last10s", "COF_whole", "COF_instrument_mean",
                  "COF_instrument_static", "Fx_offset_N", "Fx_forward_N", "Fx_return_N", "Fz_mean_N",
                  "COF_slope_end_per_s", "cof_rising_end", "n_cycles_window", "sliding_time_s", "cycles_method"):
            row[k] = c.get(k, np.nan)
        loc = pd.DataFrame(e["scans"])
        loc.insert(0, "run_id", rid)
        loc_all.append(loc)
        w = aggregate_wear(loc, "robust", p["Load_N"], S, track_mm)
        row.update({
            "wear_status": w["status"], "wear_location": w["location"], "groove_area_um2": w["area_um2"],
            "groove_LOD_um2": w["lod_um2"], "V_mm3": w["V_mm3"], "k_mm3_Nm": w["k"], "k_upper_mm3_Nm": w["k_upper"],
            "V_upper_mm3": w["V_upper_mm3"], "k_mean_loc_mm3_Nm": w["k_mean_loc"], "V_mean_loc_mm3": w["V_mean_loc_mm3"],
            "track_width_um": w["width_um"], "track_width_sd_um": w["width_sd_um"],
            "plasticity_ratio": w["plasticity_ratio"], "depth_max_um": w["depth_max_um"],
            "Sa_unworn_um": w["Sa_unworn_um"], "n_scans": w["n_scans"], "n_valid_scans": w["n_valid"],
            "n_detected_scans": w["n_detected"],
        })
        if not c.get("qc_cycles_ok", True):
            flags.append("cycles_vs_ft_gt_2pct")
        if w["n_valid"] < w["n_scans"]:
            flags.append(f"{w['n_scans'] - w['n_valid']}_scans_invalid")
        row["qc_flags"] = ";".join(flags)
        rows.append(row)
        cy = e["cycles"].copy()
        cy.insert(0, "run_id", rid)
        cyc_all.append(cy)
        # C2 wear variants
        vr = {"run_id": rid}
        for var in WEAR_VARIANTS:
            ww = aggregate_wear(loc, var, p["Load_N"], S, track_mm)
            for k in ("status", "V_mm3", "k", "k_upper", "V_mean_loc_mm3", "k_mean_loc", "width_um", "depth_max_um",
                      "n_valid", "n_detected"):
                vr[f"{var}__{k}"] = ww[k]
        ta = aggregate_wear(loc, "robust", p["Load_N"], S, track_mm, area_col="groove_ta_rule_um2")
        vr.update({"ta_rule__status": ta["status"], "ta_rule__V_mm3": ta["V_mm3"], "ta_rule__k": ta["k"]})
        var_rows.append(vr)

    runs = pd.DataFrame(rows).sort_values("run_id").reset_index(drop=True)
    runs.insert(3, "replicate_group", replicate_groups(runs).to_numpy())
    rep = runs.groupby("replicate_group")["run_id"].transform("count") > 1
    runs.insert(4, "is_replicate", rep)
    missing = sorted(set(doe.Experiment_ID) - set(runs.run_id))
    LOG.info("dataset: %d runs; DOE rows without data: %s", len(runs), ", ".join(missing))
    return {
        "runs": runs,
        "per_cycle_cof": pd.concat(cyc_all, ignore_index=True),
        "per_location_wear": pd.concat(loc_all, ignore_index=True),
        "wear_variants": pd.DataFrame(var_rows).sort_values("run_id").reset_index(drop=True),
        "missing_runs": pd.DataFrame({"run_id": missing}).merge(doe, left_on="run_id", right_on="Experiment_ID"),
    }
