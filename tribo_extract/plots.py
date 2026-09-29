"""Per-run QC figures (saved next to summary.json in reports/runs/<ID>/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e6e5e0"
HEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "height", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
    "axes.titlecolor": INK, "legend.frameon": False, "figure.dpi": 110,
})


def plot_cof(internals: dict, res: dict, title: str, out: Path):
    t, mu, roll = internals["t"], internals["mu"], internals["roll"]
    has_fz = internals.get("fz") is not None
    has_z = internals.get("z") is not None
    n = 1 + has_fz + has_z
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.3 * n + 0.4), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(t, mu, color="#c3c2b7", lw=0.8, label="COF (logged)")
    ax.plot(t, roll, color=BLUE, lw=2, label="rolling mean")
    ax.axvspan(0, internals["t_ss"], color=ORANGE, alpha=0.08, lw=0)
    ax.axvline(internals["t_ss"], color=ORANGE, lw=1.2, ls="--", label=f"steady from {internals['t_ss']:.2f} s")
    ax.axhspan(internals["ref"] - internals["band"], internals["ref"] + internals["band"], color=BLUE, alpha=0.08, lw=0)
    ax.axhline(res["COF_ss_mean"], color=BLUE, lw=1, ls=":")
    ax.set_ylabel("COF")
    ax.set_title(f"{title} — steady-state COF = {res['COF_ss_mean']:.4f} ± {res['COF_ss_std']:.4f}", loc="left")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    i = 1
    if has_fz:
        a = axes[i, 0]; i += 1
        a.plot(t, np.asarray(internals["fz"]), color=INK2, lw=0.8)
        a.set_ylabel("Fz (N)")
    if has_z:
        a = axes[i, 0]
        a.plot(t, np.asarray(internals["z"]), color=INK2, lw=0.8)
        a.set_ylabel("Z depth (µm)\n(diagnostic only)")
    axes[-1, 0].set_xlabel("sliding time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_wear(hm, internals: dict, res: dict, title: str, out: Path):
    zl, zr, v, P, g = internals["zl"], internals["zr"], internals["v"], internals["P"], internals["g"]
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1])
    ax0 = fig.add_subplot(gs[0])
    lo, hi = np.nanpercentile(zl, [1, 99])
    ny, nx = zl.shape
    im = ax0.imshow(zl, cmap=HEIGHT_CMAP, vmin=lo, vmax=hi, extent=[0, nx * res["px_um"], ny * res["px_um"], 0])
    ax0.set_title("levelled height map (grey = masked overlay)", loc="left")
    ax0.set_facecolor("#c3c2b7"); ax0.grid(False)
    cx, cy = nx * res["px_um"] / 2, ny * res["px_um"] / 2
    a = np.radians(res["track_angle_deg"]); L = max(nx, ny) * res["px_um"]
    ax0.plot([cx - L * np.cos(a), cx + L * np.cos(a)], [cy + L * np.sin(a), cy - L * np.sin(a)],
             color=ORANGE, lw=1, ls="--")
    ax0.set_xlim(0, nx * res["px_um"]); ax0.set_ylim(ny * res["px_um"], 0)
    ax0.set_xlabel("x (µm)"); ax0.set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax0, fraction=0.035, label="height (µm)")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(zr, cmap=HEIGHT_CMAP, vmin=lo, vmax=hi, aspect="auto")
    ax1.set_facecolor("#c3c2b7"); ax1.grid(False)
    ax1.set_title(f"rotated so sliding is horizontal ({res['track_angle_deg']:.1f}°)", loc="left")
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[2])
    ref = g["ref"]
    ax2.plot(v, P, color="#c3c2b7", lw=0.8, label="mean profile")
    ax2.plot(v, g["Ps"], color=BLUE, lw=1.6, label="smoothed")
    ax2.plot(v, ref, color=INK2, lw=1, ls="--", label="reference (shoulders)")
    if res.get("wear_detected"):
        L_, R_ = g["L"], g["R"]
        ax2.fill_between(v[L_:R_ + 1], g["Ps"][L_:R_ + 1], ref[L_:R_ + 1], color=ORANGE, alpha=0.35,
                         lw=0, label=f"worn area {res['worn_area_um2']:.1f} µm²")
        head = f"width {res['track_width_um']:.0f} µm, depth {res['track_max_depth_um']:.2f} µm"
    else:
        head = f"no wear track (LOD ≈ {res['area_LOD_um2']:.1f} µm²)"
    ax2.set_title(head, loc="left")
    ax2.set_xlabel("across-track position (µm)"); ax2.set_ylabel("height (µm)")
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
