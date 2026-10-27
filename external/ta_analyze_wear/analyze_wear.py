"""
Unified Standalone RTEC Tribometer & Profilometry Analysis Pipeline
Author: AI Collaborator for Angel Singhvi (IIT Bombay ME323)
"""

import os
import glob
import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from sklearn.linear_model import HuberRegressor

# ==========================================
# 1. BCRF 3D HEIGHT MAP INGESTION
# ==========================================
def read_bcrf(path):
    """Parse RTEC .bcrf height map: 4096-byte UTF-16 header + float32 matrix."""
    with open(path, "rb") as f:
        raw = f.read()

    head = raw[:4096].decode("utf-16le", errors="ignore").replace("\x00", "")
    meta = {}
    for line in head.splitlines():
        if "=" in line:
            k, v = [s.strip() for s in line.split("=", 1)]
            meta[k] = v

    nx, ny = int(meta["xpixels"]), int(meta["ypixels"])
    z = (
        np.frombuffer(raw[4096 : 4096 + nx * ny * 4], dtype="<f4")
        .reshape(ny, nx)
        .astype(float)
    )
    z[z > 1e30] = np.nan  # Instrument void values
    px_um = float(meta["xlength"]) / nx / 1000.0  # nm -> um per pixel
    return z, px_um, meta

# ==========================================
# 2. RTEC FRICTION CSV LOG LOADER
# ==========================================
def read_friction_csv(path):
    """Robust RTEC friction log loader that skips metadata headers."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = 0
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if "step" in line_lower and ("time" in line_lower or "t," in line_lower or ",t," in line_lower):
            if "uppersample" not in line_lower and "customer" not in line_lower:
                header_idx = idx
                break

    df = pd.read_csv(io.StringIO("".join(lines[header_idx:])), skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "step" or cl.startswith("step"):
            col_map[col] = "step"
        elif cl in ["t", "time", "timestamp"] or "time" in cl:
            if "t" not in col_map.values():
                col_map[col] = "t"
        elif "fz" in cl or "normal" in cl:
            if "cof" not in cl:
                col_map[col] = "Fz"
        elif "daq.fx" in cl or cl == "fx":
            if "cof" not in cl:
                col_map[col] = "Fx"
        elif "cof" in cl:
            col_map[col] = "COF"

    df = df.rename(columns=col_map)

    if "step" not in df.columns:
        df["step"] = df.iloc[:, 0]
    if "t" not in df.columns:
        df["t"] = df.iloc[:, 1]
    if "Fz" not in df.columns:
        df["Fz"] = df.iloc[:, 3]
    if "Fx" not in df.columns:
        df["Fx"] = df.iloc[:, 5]

    for col in ["t", "Fz", "Fx"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "COF" in df.columns:
        df["COF"] = pd.to_numeric(df["COF"], errors="coerce")

    df["step"] = df["step"].astype(str).str.strip()
    df = df.dropna(subset=["t", "Fz", "Fx"])
    return df

# ==========================================
# 3. IMAGE PREPROCESSING & LEVELING
# ==========================================
def inpaint_voids(z):
    z_clean = z.copy()
    invalid = ~np.isfinite(z_clean)
    if not np.any(invalid):
        return z_clean
    valid = np.isfinite(z_clean)
    indices = np.indices(z_clean.shape)
    coords_val = np.column_stack((indices[0][valid], indices[1][valid]))
    coords_inv = np.column_stack((indices[0][invalid], indices[1][invalid]))
    z_clean[invalid] = griddata(coords_val, z_clean[valid], coords_inv, method="nearest")
    return z_clean

def orient_across_track(z):
    std_y = np.nanstd(np.nanmean(z, axis=0))
    std_x = np.nanstd(np.nanmean(z, axis=1))
    return z.T if std_y > std_x else z

def robust_level(z, order=1, shoulder_fraction=0.15):
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    X, Y = (xx - nx / 2.0) / nx, (yy - ny / 2.0) / ny
    terms = [X**i * Y**j for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack(terms, axis=-1)

    edge_px = int(ny * shoulder_fraction)
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:edge_px, :] = True
    mask[-edge_px:, :] = True
    mask = mask & np.isfinite(z)

    for _ in range(3):
        if np.sum(mask) < (order + 1) * (order + 2):
            break
        c, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
        res = z - A @ c
        std = np.nanstd(res[mask])
        mask = mask & (np.abs(res - np.nanmedian(res[mask])) < 2.0 * std)

    c, *_ = np.linalg.lstsq(A[mask], z[mask], rcond=None)
    fitted_surface = A @ c
    return z - fitted_surface, mask

# ==========================================
# 4. METRICS & GEOMETRY EXTRACTION
# ==========================================
def compute_groove_metrics(zl, px_um, mask_unworn, track_len_mm=10.0):
    prof = np.nanmean(zl, axis=1)
    
    window_len = min(51, len(prof) // 4 * 2 + 1)
    if window_len > 5:
        prof = savgol_filter(prof, window_length=window_len, polyorder=3)

    shoulder_rows = np.where(np.any(mask_unworn, axis=1))[0]
    base = np.median(prof[shoulder_rows]) if len(shoulder_rows) else np.median(prof)
    base_sd = np.std(prof[shoulder_rows]) if len(shoulder_rows) else np.std(prof)

    d = base - prof  # Positive = wear depth, Negative = pile-up
    k_max = int(np.nanargmax(d))
    dmax = float(d[k_max])

    left_edge = 0
    for i in range(k_max, -1, -1):
        if d[i] <= 0:
            left_edge = i
            break

    right_edge = len(d) - 1
    for i in range(k_max, len(d)):
        if d[i] <= 0:
            right_edge = i
            break

    width_um = float((right_edge - left_edge) * px_um)
    scar_slice = slice(left_edge, right_edge + 1)

    cutoff = max(2.5 * base_sd, 0.25)
    groove_loss = np.clip(d[scar_slice], 0, None)
    groove_loss[groove_loss < cutoff] = 0.0

    area_groove = float(groove_loss.sum() * px_um)
    area_pileup = float(np.clip(-d[:left_edge], 0, None).sum() * px_um) + float(
        np.clip(-d[right_edge:], 0, None).sum() * px_um
    )

    metrics = {
        "depth_max_um": dmax,
        "width_um": width_um,
        "area_groove_um2": area_groove,
        "area_pileup_um2": area_pileup,
        "plasticity_ratio": (area_pileup / area_groove if area_groove > 0 else 0.0),
        "vol_track_mm3": float(area_groove * 1e-6 * track_len_mm),
        "baseline_sd_um": float(base_sd),
    }
    return metrics, prof, base

def compute_friction_features(df):
    step_id = "2" if "2" in df["step"].values else df["step"].value_counts().index[0]
    s = df[df.step == step_id].copy()
    if len(s) == 0:
        s = df.copy()

    t_min = s["t"].min() if len(s) > 0 else 0.0
    s = s[s.t > t_min + 1.0]
    if len(s) == 0:
        s = df.copy()

    s["sec"] = (s["t"] // 1).astype(int)

    if "COF" in s.columns:
        s["cof"] = pd.to_numeric(s["COF"], errors="coerce")
    elif "DAQ.COF (Fx)" in s.columns:
        s["cof"] = pd.to_numeric(s["DAQ.COF (Fx)"], errors="coerce")
    else:
        fx = pd.to_numeric(s["Fx"], errors="coerce").abs()
        fz = pd.to_numeric(s["Fz"], errors="coerce")
        s["cof"] = fx / fz

    fz = pd.to_numeric(s["Fz"], errors="coerce")
    temp_df = pd.DataFrame({"sec": s["sec"], "cof": s["cof"]})
    cof_mean = temp_df.groupby("sec")["cof"].mean().dropna()

    if len(cof_mean) == 0:
        val = float(s["cof"].mean()) if len(s) > 0 else 0.0
        cof_mean = pd.Series([val], index=[0])

    initial_mean = float(cof_mean.iloc[:10].mean()) if len(cof_mean) > 0 else 0.0
    steady_mean = float(cof_mean.iloc[-10:].mean()) if len(cof_mean) > 0 else 0.0
    overall_mean = float(cof_mean.mean()) if len(cof_mean) > 0 else 0.0

    fric_metrics = {
        "Fz_mean": float(fz.mean()) if len(fz) > 0 else 0.0,
        "Fz_sd": float(fz.std()) if len(fz) > 0 else 0.0,
        "cof_initial": initial_mean,
        "cof_steady_state": steady_mean,
        "cof_overall_mean": overall_mean,
    }
    return fric_metrics, cof_mean

# ==========================================
# 5. ORCHESTRATION PIPELINE & PLOTTING
# ==========================================
def run_pipeline(exp_dir, load_N=10.0, freq_Hz=6.0, stroke_mm=5.0, time_s=120.0, target_scan=None):
    csv_files = glob.glob(os.path.join(exp_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No friction log CSV found inside: {exp_dir}")
    friction_csv = csv_files[0]
    print(f"Using friction log: {os.path.basename(friction_csv)}")

    track_len_mm = 2.0 * stroke_mm
    bcrf_files = sorted(
        glob.glob(os.path.join(exp_dir, "*.bcrf")),
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
    )

    if not bcrf_files:
        raise FileNotFoundError(f"No .bcrf scans located inside: {exp_dir}")

    print(f"Found {len(bcrf_files)} scans. Analyzing individual profiles...")

    scan_results = []
    profiles = []

    for path in bcrf_files:
        z, px_um, meta = read_bcrf(path)
        z_clean = inpaint_voids(orient_across_track(z))
        zl, mask = robust_level(z_clean, order=1, shoulder_fraction=0.15)

        m, prof, base = compute_groove_metrics(zl, px_um, mask, track_len_mm=track_len_mm)
        m["filename"] = os.path.basename(path)
        scan_results.append(m)
        profiles.append((prof, base, os.path.basename(path)))

    df_scans = pd.DataFrame(scan_results)
    df_scans.to_csv(os.path.join(exp_dir, "all_scans_breakdown.csv"), index=False)

    if target_scan:
        match_idx = df_scans[df_scans["filename"].str.contains(target_scan, case=False)].index
        selected_idx = match_idx[0] if len(match_idx) > 0 else df_scans["depth_max_um"].idxmax()
    else:
        selected_idx = df_scans["depth_max_um"].idxmax()

    best_scan_metrics = df_scans.loc[selected_idx]
    selected_profile, selected_base, selected_name = profiles[selected_idx]

    print("\n" + "=" * 50)
    print(f"SELECTED SCAN FOR REPORTING: {selected_name}")
    print("=" * 50)
    print(f" - Max Wear Depth:     {best_scan_metrics['depth_max_um']:.2f} um")
    print(f" - Track Width:        {best_scan_metrics['width_um']:.1f} um")
    print(f" - Groove Area:        {best_scan_metrics['area_groove_um2']:.1f} um^2")
    print(f" - Track Volume:       {best_scan_metrics['vol_track_mm3']:.5e} mm^3")
    print(f" - Plasticity Ratio:   {best_scan_metrics['plasticity_ratio']:.3f}")
    print("=" * 50)

    df_fric = read_friction_csv(friction_csv)
    fric_metrics, cof_series = compute_friction_features(df_fric)

    sliding_dist_m = 4.0 * (stroke_mm / 1000.0) * freq_Hz * time_s
    k_archard = best_scan_metrics["vol_track_mm3"] / (load_N * sliding_dist_m)

    summary = {
        "run_id": os.path.basename(exp_dir),
        "selected_scan": selected_name,
        "load_N": load_N,
        "freq_Hz": freq_Hz,
        "stroke_mm": stroke_mm,
        "sliding_dist_m": sliding_dist_m,
        "depth_max_um": best_scan_metrics["depth_max_um"],
        "width_um": best_scan_metrics["width_um"],
        "vol_track_mm3": best_scan_metrics["vol_track_mm3"],
        "k_archard_mm3_per_Nm": k_archard,
        **fric_metrics,
    }

    pd.DataFrame([summary]).to_csv(os.path.join(exp_dir, "single_scan_summary.csv"), index=False)
    print(f"Saved summary to: {os.path.join(exp_dir, 'single_scan_summary.csv')}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(selected_profile - selected_base, label=f"Selected: {selected_name}", color="tab:red", lw=2)
    ax1.axhline(0, color="k", linestyle="--", alpha=0.5, label="Baseline")
    ax1.set_title(f"Wear Profile ({selected_name})")
    ax1.set_xlabel("Pixel index across track")
    ax1.set_ylabel("Depth relative to baseline (um)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(cof_series.index, cof_series.values, color="tab:blue", lw=1.2)
    ax2.set_title("Friction Coefficient vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("COF (Fx / Fz)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    target_dir = r"C:\Users\singh\OneDrive\Desktop\me323projectwork\R01"
    run_pipeline(
        target_dir, 
        load_N=10.0, 
        freq_Hz=6.0, 
        stroke_mm=5.0, 
        time_s=120.0, 
        target_scan="R01b_13"
    )