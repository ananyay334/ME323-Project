"""The presentation figures, numbered in slide order. One function per figure; each returns (caption, source).

Every figure is listed in outputs/figures/INDEX.md with its caption and data source.
"""

from __future__ import annotations

import traceback
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..config import Paths
from ..evaluate.metrics import r2, tercile_confusion
from ..extract.cof import detect_reversals, find_sliding_step
from ..extract.wear_band import analyse_scan
from ..io.bcrf import read_bcrf
from ..io.csv_reader import read_friction_csv
from ..io.layout import find_runs, load_doe
from ..models.physics import LinearPhysics
from ..models.registry import MODEL_ORDER, MODELS, TARGETS, model_label, target_frame
from ..utils import LOG
from . import style as S
from .style import INK, INK2, MUTED, SURFACE, save, title

TLABEL = {k: t.short for k, t in TARGETS.items()}


class Data:
    def __init__(self, cfg: dict, paths: Paths):
        self.cfg, self.paths = cfg, paths

    def t(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.paths.tables / f"{name}.csv")

    def d(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.paths.data / f"{name}.csv")

    @cached_property
    def runs(self):
        return self.d("runs")

    @cached_property
    def oof(self):
        return self.d("oof_predictions")

    @cached_property
    def met(self):
        return self.t("metrics_all")

    @cached_property
    def sel(self):
        return self.t("selection").set_index("target")

    @cached_property
    def folders(self):
        return {r.folder_id: r for r in find_runs(self.paths.testdata)}

    def run_files(self, run_id: str):
        fid = self.runs.set_index("run_id").at[run_id, "folder_id"]
        return self.folders[fid]

    def final(self, target: str) -> str:
        return self.sel.at[target, "final_model"]

    def selected_ml(self, target: str) -> str:
        s = self.sel.at[target, "selected_ml"]
        return s if isinstance(s, str) and s else ""


def _fig(rows=1, cols=1, top=0.86, **kw):
    fig, axes = plt.subplots(rows, cols, figsize=S.SLIDE, **kw)
    fig.subplots_adjust(top=top, bottom=0.1, left=0.07, right=0.98, wspace=0.28, hspace=0.42)
    return fig, axes


def _plain_log(ax, axis="both"):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    fmt = FuncFormatter(lambda x, _: f"{x:g}")
    for a in (("x", "y") if axis == "both" else (axis,)):
        getattr(ax, f"{a}axis").set_major_formatter(fmt)
        getattr(ax, f"{a}axis").set_minor_formatter(NullFormatter())


def _r2txt(ax, y, p, loc=(0.03, 0.95)):
    ax.text(*loc, f"R² = {r2(np.asarray(y), np.asarray(p)):+.2f}", transform=ax.transAxes, va="top", fontsize=12, color=INK)


# ============================================================================================ 1 pipeline
def fig01_pipeline(D: Data):
    fig = plt.figure(figsize=S.SLIDE)
    title(fig, "From raw tribometer files to validated COF and wear models",
          "one command (python run_all.py) runs every stage; each stage caches its outputs")
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.85])
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    boxes = [
        ("Raw files", "23 run folders\n.csv friction log (100 Hz)\n.lrx recipe\n.bcrf height maps (4/run)\n.Raw / .rtec never opened", S.MUTED),
        ("Extraction", "COF per cycle\n(P90-P10)/2 / Fz\n\nwear: smooth band\nrobust levelling, LOD\nmost-worn location", S.BLUE),
        ("Dataset", "runs.csv: 1 row / run\nL, f, t, S, v\nCOF targets\nV, k, width, status\nQC flags", S.AQUA),
        ("Models", "physics laws\nANN, GPR, TabPFN\nhybrids (physics + ML)\nconstant, power law,\nRidge, SVR, XGBoost", S.ORANGE),
        ("Evaluation", "P1: LOGO, 23 runs\nP2: 18 training runs\nP3: 5 test runs\nmetrics, 1-SE ties,\npermutation tests", S.VIOLET),
        ("Outputs", "22 figures\ntables (CSV)\nPDF report\nslide numbers\nREADME", S.INK),
    ]
    w, h, gap = 15.0, 40.0, 1.6
    for i, (head, body, c) in enumerate(boxes):
        x = 1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 8), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=SURFACE, ec=c, lw=2.2))
        ax.add_patch(Rectangle((x - 0.4, 8 + h - 7), w + 0.8, 7.4, fc=c, ec=c))
        ax.text(x + w / 2, 8 + h - 3.3, head, ha="center", va="center", color="white", fontsize=15, fontweight="bold")
        ax.text(x + w / 2, 8 + (h - 7) / 2, body, ha="center", va="center", color=INK, fontsize=11.5, linespacing=1.5)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, 28), (x + w + gap - 0.5, 28), arrowstyle="-|>", mutation_scale=18,
                                         color=INK2, lw=1.6))
    ax.text(50, 2.5, "The test day is known (13 / 16 Oct) but is never a model input or correction.", ha="center",
            fontsize=12, color=INK2)
    return "Pipeline overview: raw files -> extraction -> dataset -> models -> evaluation -> outputs.", "code structure (src/tribo_final)"


# ============================================================================================ 2 test matrix
def fig02_test_matrix(D: Data):
    runs = D.runs
    doe = load_doe(D.paths.inputs / "run_sheet.csv")
    missing = doe[~doe.Experiment_ID.isin(runs.run_id)]
    fig, axes = _fig(1, 2, top=0.84)
    title(fig, f"{len(runs)} of the 30 planned runs exist; load spans 10-100 N, frequency 4-15 Hz, time 60-120 s",
          "filled = run with data (blue training, orange test), ring = replicate R04/R14/R20, hollow grey = planned but missing")
    for ax, (xc, xl) in zip(axes, (("Freq_Hz", "frequency f (Hz)"), ("Run_time_s", "run time t (s)"))):
        for role, (c, m) in S.ROLE_STYLE.items():
            d = runs[runs.role == role]
            ax.scatter(d[xc], d.Load_N, s=90, marker=m, color=c, edgecolors=SURFACE, lw=1, label=f"{role} ({len(d)})", zorder=3)
        rep = runs[runs.is_replicate]
        ax.scatter(rep[xc], rep.Load_N, s=420, facecolors="none", edgecolors=INK, lw=1.4, label="replicates (3 runs, 1 point)", zorder=4)
        ax.scatter(missing[xc], missing.Load_N, s=90, facecolors="none", edgecolors=MUTED, lw=1.5, label=f"missing ({len(missing)})", zorder=2)
        for _, r in missing.iterrows():
            ax.annotate(r.Experiment_ID, (r[xc], r.Load_N), xytext=(6, -4), textcoords="offset points", fontsize=10, color=MUTED)
        for _, r in runs[runs.role == "test"].iterrows():
            ax.annotate(r.run_id, (r[xc], r.Load_N), xytext=(7, 3), textcoords="offset points", fontsize=10, color=INK2)
        ax.set_yscale("log")
        ax.set_yticks([10, 20, 50, 100])
        ax.set_yticklabels(["10", "20", "50", "100"])
        ax.set_xlabel(xl)
        ax.set_ylabel("load L (N)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=12)
    fig.subplots_adjust(bottom=0.18)
    return "Test matrix of the 23 runs in L-f-t space; replicates ringed, missing DOE rows hollow.", "outputs/data/runs.csv, inputs/run_sheet.csv"


# ============================================================================================ 3 raw signal
def fig03_raw_signal(D: Data):
    rf = D.run_files("R04")
    log = read_friction_csv(rf.csv)
    df = log.data
    cc = D.cfg["cof"]
    step = find_sliding_step(df, cc)
    t, fx, fz = df.t.to_numpy(), df.Fx.to_numpy(), df.Fz.to_numpy()
    ts, te = t[step.start], t[step.stop - 1]
    fig = plt.figure(figsize=S.SLIDE)
    title(fig, "R04 raw signal: the per-cycle half-range of Fx cancels the sensor offset",
          "COF per cycle = (P90 - P10)/2 / mean Fz; the instrument |Fx|/Fz column is biased by the offset and the reversals")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], left=0.07, right=0.98, top=0.85, bottom=0.17, hspace=0.4, wspace=0.22)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for ax in (a1, a2):
        ax.axvspan(t[0], ts, color=S.GRID, alpha=0.9, lw=0)
        ax.axvspan(ts, ts + cc["ramp_trim_s"], color="#f3e1d6", lw=0)
        ax.axvspan(te - cc["ramp_trim_s"], te, color="#f3e1d6", lw=0)
    a1.plot(t, fz, color=INK, lw=0.6)
    a1.set_ylabel("Fz (N)")
    a1.set_title("Fz: static step (grey) and the 1 s ramps (orange) are excluded", fontsize=12)
    a2.plot(t, fx, color=S.BLUE, lw=0.4)
    off = float(D.runs.set_index("run_id").at["R04", "Fx_offset_N"])
    a2.axhline(off, color=S.RED, lw=1.4)
    a2.set_title(f"Fx: forward and return strokes about an offset of {off:+.2f} N (red line)", fontsize=12)
    a2.set_ylabel("Fx (N)")
    a2.set_xlabel("time (s)")
    # zoom: three cycles in the middle
    mid = ts + 40
    m = (t >= mid) & (t <= mid + 3 / 7.5)
    seg = slice(np.flatnonzero(m)[0], np.flatnonzero(m)[-1] + 1)
    a3.plot(t[seg], fx[seg], color=S.BLUE, marker="o", ms=4, lw=1.2, label="Fx (100 Hz samples)")
    rev, direc, info = detect_reversals(fx[step.start:step.stop], log.fs, cc)
    rv = rev + step.start
    rv = rv[(t[rv] >= t[seg][0]) & (t[rv] <= t[seg][-1])]
    a3.scatter(t[rv], fx[rv], s=70, facecolors="none", edgecolors=INK, lw=1.4, zorder=4, label="detected reversals")
    q10, q90 = np.percentile(fx[seg], [10, 90])
    a3.axhline(q90, color=INK2, lw=1)
    a3.axhline(q10, color=INK2, lw=1)
    a3.axhline((q10 + q90) / 2, color=S.RED, lw=1.2, label="offset (P10+P90)/2")
    a3.annotate("", (t[seg][-1] + 0.01, q90), (t[seg][-1] + 0.01, q10), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    a3.text(t[seg][-1] - 0.005, (q10 + q90) / 2 + 0.6, f"P90-P10\n= {q90 - q10:.2f} N", ha="right", fontsize=11)
    a3.text(t[seg][0], q90 + 0.3, f"COF = {(q90 - q10) / 2 / fz[seg].mean():.3f}", fontsize=12, color=INK)
    a3.set_title("zoom: three cycles at 7.5 Hz")
    a3.set_xlabel("time (s)")
    a3.set_ylabel("Fx (N)")
    a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10)
    return ("R04 raw Fz and Fx: static step, trimmed ramps, Fx offset, reversals and the per-cycle half-range COF.",
            "testdata/R04/R04.csv")


# ============================================================================================ 4 COF vs time
def fig04_cof_time(D: Data):
    runs = D.runs.sort_values(["Load_N", "run_id"])
    cyc = D.d("per_cycle_cof")
    n_rise = int(runs.cof_rising_end.sum())
    fig, axes = plt.subplots(4, 6, figsize=S.SLIDE, sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.99, wspace=0.08, hspace=0.45)
    title(fig, f"COF is still rising at the end in {n_rise} of {len(runs)} runs: no run reaches steady state",
          "per-cycle COF (dots) and 15-cycle running median, ordered by load; shaded = first 10 s of the trimmed window (primary target)")
    axes = axes.ravel()
    w0 = D.cfg["cof"]["ramp_trim_s"]
    for ax, (_, r) in zip(axes, runs.iterrows()):
        c = cyc[cyc.run_id == r.run_id]
        ax.axvspan(w0, w0 + D.cfg["cof"]["first_window_s"], color="#e3eefb", lw=0)
        ax.scatter(c.t_s, c.cof, s=2, color=S.BLUE_RAMP[2], lw=0, rasterized=True)
        ax.plot(c.t_s, c.cof.rolling(15, center=True, min_periods=5).median(), color=INK, lw=1.4)
        ax.set_title(f"{r.run_id}  {r.Load_N:g} N, {r.Freq_Hz:g} Hz", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_ylim(0.08, 0.2)
    for ax in axes[len(runs):]:
        ax.set_axis_off()
    for ax in axes[::6]:
        ax.set_ylabel("COF", fontsize=11)
    for ax in axes[18:23]:
        ax.set_xlabel("sliding time (s)", fontsize=11)
    return ("COF vs sliding time for all 23 runs (small multiples by load); first-10-s window shaded.",
            "outputs/data/per_cycle_cof.csv")


# ============================================================================================ 5 height map
def _scan(D: Data, run_id: str, loc: str):
    rf = D.run_files(run_id)
    s = [x for x in rf.scans if x.location == loc][0]
    return s, read_bcrf(s.bcrf)


def fig05_height_map(D: Data):
    s, hm = _scan(D, "R04", "m")
    r = analyse_scan(hm.z, hm.pixel_um, D.cfg["wear"], keep_arrays=True)
    A = r.arrays
    y_um = np.arange(hm.ny) * hm.pixel_um
    fig = plt.figure(figsize=S.SLIDE)
    st = "detected" if r.detected else "below the LOD"
    title(fig, f"R04 (m): the worn track is the smooth band; its groove ({r.groove_um2:.0f} µm²) is {st} (LOD {r.lod_um2:.0f} µm²)",
          f"track width {r.width_um:.0f} µm (core); band = core ± 40 % tightened to the ridge feet; reference = rows outside band + 10 µm")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.45, 1.25], left=0.06, right=0.98, top=0.85, bottom=0.2, wspace=0.12)
    a0 = fig.add_subplot(gs[0])
    a1, a2 = fig.add_subplot(gs[1], sharey=a0), fig.add_subplot(gs[2])
    z2 = A["z2"]
    lim = np.nanpercentile(np.abs(z2), 98)
    im = a0.imshow(z2, aspect="auto", cmap=S.DIVERGING, vmin=-lim, vmax=lim,
                   extent=[0, hm.nx * hm.pixel_um, y_um[-1], 0], interpolation="nearest")
    a0.grid(False)
    for (lo, hi), c, lab in ((r.core, S.INK, "core"), (r.band, S.ORANGE, "band")):
        for v in (lo, hi):
            a0.axhline(v * hm.pixel_um, color=c, lw=1.6)
        a0.text(hm.nx * hm.pixel_um * 0.99, lo * hm.pixel_um, lab, ha="right", va="bottom", color=c, fontsize=11,
                fontweight="bold", bbox=dict(fc=SURFACE, ec="none", pad=1))
    a0.set_xlabel("along the track x (µm)")
    a0.set_ylabel("across the track y (µm)")
    cax = a0.inset_axes([0.0, -0.2, 1.0, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("height vs reference (µm)")
    a1.plot(A["rough"], y_um, color=INK, lw=1.4)
    a1.axvline(r.R_unworn, color=MUTED, lw=1)
    a1.axvline(r.R_track, color=S.BLUE, lw=1)
    a1.axhspan(r.core[0] * hm.pixel_um, r.core[1] * hm.pixel_um, color="#e3eefb", lw=0)
    a1.tick_params(labelleft=False)
    a1.set_xlabel("row roughness (µm)")
    a1.set_title("smooth = worn", fontsize=12)
    p = A["profile"]
    ref = np.zeros(hm.ny, bool)
    ref[A["ref_rows"]] = True
    a2.axvspan(r.band[0] * hm.pixel_um, r.band[1] * hm.pixel_um, color="#fbeee7", lw=0, label="band")
    a2.plot(y_um, p, color=INK, lw=1.4, label="across-track profile")
    band = np.zeros(hm.ny, bool)
    band[r.band[0]:r.band[1]] = True
    a2.fill_between(y_um, p, 0, where=band & (p < 0), color=S.BLUE, alpha=0.5, lw=0, label=f"groove {r.groove_um2:.0f} µm²")
    a2.fill_between(y_um, p, 0, where=band & (p > 0), color=S.RED, alpha=0.45, lw=0, label=f"pile-up {r.pileup_um2:.0f} µm²")
    a2.scatter(y_um[ref], np.full(ref.sum(), np.nanmin(p) - 0.4), s=2, color=MUTED, label="reference rows")
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_xlabel("across the track y (µm)")
    a2.set_ylabel("mean height (µm)")
    a2.legend(loc="upper right", fontsize=10)
    return ("Height map R04 (m) with core, band and reference rows; across-track profile with groove and pile-up shading.",
            f"testdata/R04/{s.bcrf.name}")


# ============================================================================================ 6 robust levelling
def fig06_robust(D: Data):
    s, hm = _scan(D, "R24", "m")
    rr = analyse_scan(hm.z, hm.pixel_um, D.cfg["wear"], robust=True, keep_arrays=True)
    rp = analyse_scan(hm.z, hm.pixel_um, D.cfg["wear"], robust=False, keep_arrays=True)
    y_um = np.arange(hm.ny) * hm.pixel_um
    fig = plt.figure(figsize=S.SLIDE)
    pit = np.nanmin(rr.arrays["z1"]) - np.nanmedian(rr.arrays["z1"])
    title(fig, "Robust levelling ignores the deep pit that drags a plain least-squares fit",
          f"R24 location m: a pit {abs(pit):.0f} µm below the median surface; 2σ clipping (3 passes) drops it from the plane and the row means")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.06, right=0.98, top=0.85, bottom=0.11, wspace=0.3)
    a0, a1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    z1 = rr.arrays["z1"]
    lim = np.nanpercentile(np.abs(z1), 99)
    im = a0.imshow(z1, aspect="auto", cmap=S.DIVERGING, vmin=-lim, vmax=lim, extent=[0, hm.nx * hm.pixel_um, y_um[-1], 0])
    a0.grid(False)
    iy, ix = np.unravel_index(np.nanargmin(z1), z1.shape)
    a0.annotate("pit", (ix * hm.pixel_um, iy * hm.pixel_um), xytext=(30, -30), textcoords="offset points", color=INK,
                fontsize=12, arrowprops=dict(arrowstyle="->", color=INK))
    fig.colorbar(im, ax=a0, fraction=0.04, pad=0.02).set_label("height after first levelling (µm)")
    a0.set_xlabel("along the track x (µm)")
    a0.set_ylabel("across the track y (µm)")
    if "profile" in rp.arrays:
        a1.plot(y_um, rp.arrays["profile"], color=MUTED, lw=1.6, label="plain fit, plain row mean")
    if "profile" in rr.arrays:
        a1.plot(y_um, rr.arrays["profile"], color=S.BLUE, lw=1.8, label="robust fit, 2σ-clipped row mean")
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xlabel("across the track y (µm)")
    a1.set_ylabel("mean height (µm)")
    a1.legend(loc="lower left")
    return "Robust vs plain levelling on R24 (m), a scan with a deep pit.", f"testdata/R24/{s.bcrf.name}"


# ============================================================================================ 7 wear per run
def fig07_wear_per_run(D: Data):
    runs = D.runs
    kc = D.t("k_material_constant").iloc[0]
    fig, ax = _fig(top=0.84)
    pl = D.t("power_law_exponents").set_index(["target", "input"])
    lo, hi = pl.at[("log10_k", "lnL"), "ci_lo"], pl.at[("log10_k", "lnL"), "ci_hi"]
    trend = "no detectable trend in load: one Archard constant" if lo < 0 < hi else "a load trend (exponent CI excludes 0)"
    title(fig, f"k ≈ {kc.k_geo_mean:.1e} mm³/(N·m) ×/÷ {kc.k_gsd:.1f}, with {trend}",
          "most-worn valid location per run; below-LOD runs shown as upper bounds; not-measurable runs marked at the bottom")
    det = runs[runs.wear_status == "detected"]
    low = runs[runs.wear_status == "below_LOD"]
    nm = runs[runs.wear_status == "not_measurable"]
    ax.axhspan(kc.k_geo_mean / kc.k_gsd, kc.k_geo_mean * kc.k_gsd, color="#eceae4", lw=0, label="geometric mean ×/÷ GSD")
    ax.axhline(kc.k_geo_mean, color=INK, lw=1.4)
    ax.scatter(det.Load_N, det.k_mm3_Nm, s=80, color=INK, edgecolors=SURFACE, zorder=3, label=f"detected ({len(det)})")
    ax.scatter(low.Load_N, low.k_upper_mm3_Nm, s=80, marker="v", facecolors=SURFACE, edgecolors=S.ORANGE, lw=1.8, zorder=3,
               label=f"below LOD: upper bound ({len(low)})")
    ymin = np.nanmin(np.r_[det.k_mm3_Nm, low.k_upper_mm3_Nm]) / 2.5
    ax.scatter(nm.Load_N, np.full(len(nm), ymin), s=90, marker="x", color=S.RED, lw=2, zorder=3,
               label=f"not measurable ({len(nm)})")
    for d in (det, low, nm):
        for _, r in d.iterrows():
            yv = r.k_mm3_Nm if r.wear_status == "detected" else (r.k_upper_mm3_Nm if r.wear_status == "below_LOD" else ymin)
            ax.annotate(r.run_id, (r.Load_N, yv), xytext=(6, 2), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_xlabel("load L (N)")
    ax.set_ylabel("specific wear rate k (mm³/(N·m))")
    ax.legend(loc="upper left", ncol=2)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    return "Specific wear rate k per run vs load with below-LOD upper bounds and not-measurable runs.", "outputs/data/runs.csv"


# ============================================================================================ 8 C1
def fig08_c1(D: Data):
    c1 = D.t("c1_cof_definitions")
    fig, axes = _fig(1, 2, top=0.84)
    fig.subplots_adjust(left=0.2)
    f10 = c1.set_index("definition").loc["COF_first10s"]
    title(fig, f"The first-10-s COF follows the adhesion law best (R² P1 {f10.physics_R2_P1:+.2f}, P2 {f10.physics_R2_P2:+.2f})",
          f"six COF definitions: replicate scatter and the physics-law R² of COF = μ₀ + c·L^(-1/3); the instrument mean is "
          f"{100 * (1 - c1.instrument_ratio_mean.iloc[0]):.0f} % low and its ratio to ours falls with f (r = {c1.instrument_ratio_vs_f_r.iloc[0]:+.2f})")
    yy = np.arange(len(c1))[::-1]
    col = [S.BLUE if d == "COF_first10s" else MUTED for d in c1.definition]
    axes[0].barh(yy, c1.replicate_sd, color=col, height=0.6)
    for y, v, snr in zip(yy, c1.replicate_sd, c1.signal_to_noise):
        axes[0].text(v, y, f"  {v:.4f}  (S/N {snr:.1f})", va="center", fontsize=11, color=INK2)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(c1.label)
    axes[0].set_xlabel("replicate SD of COF (R04, R14, R20)")
    axes[0].set_xlim(0, c1.replicate_sd.max() * 1.8)
    axes[0].set_title("noise floor")
    ax = axes[1]
    ax.axvline(0, color=AXISC, lw=1)
    ax.scatter(c1.physics_R2_P1, yy, s=90, color=INK, label="P1 (23 runs)", zorder=3)
    ax.scatter(c1.physics_R2_P2, yy, s=90, facecolors=SURFACE, edgecolors=INK, lw=1.8, label="P2 (18 training runs)", zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_xlabel("cross-validated R² of μ₀ + c·L^(-1/3)")
    ax.set_title("physics-law fit")
    ax.legend(loc="upper left")
    return "C1: COF definitions compared by replicate SD and physics-law R² (P1, P2).", "outputs/tables/c1_cof_definitions.csv"


AXISC = S.AXIS


# ============================================================================================ 9 C2
def fig09_c2(D: Data):
    c2 = D.t("c2_wear_measures")
    fig, axes = _fig(1, 3, top=0.84)
    a = c2.set_index("measure").loc["V_maxloc_robust"]
    title(fig, f"Most-worn location + robust levelling: {int(a.runs_measurable)} runs with an LOD, replicate SD {a.replicate_sd_dex:.2f} dex",
          "wear measures compared: runs measurable, replicate scatter (log10) and the best cross-validated R² (Archard or power law)")
    yy = np.arange(len(c2))[::-1]
    col = [S.BLUE if m == "V_maxloc_robust" else MUTED for m in c2.measure]
    axes[0].barh(yy, c2.runs_measurable, color=col, height=0.6)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([l.replace(" (", "\n(").replace(", ", ",\n", 1) if len(l) > 34 else l for l in c2.label], fontsize=10)
    axes[0].set_xlabel("runs measurable (of 23)")
    axes[1].barh(yy, c2.replicate_sd_dex, color=col, height=0.6)
    axes[1].set_yticks(yy)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("replicate SD (dex)")
    axes[2].axvline(0, color=AXISC, lw=1)
    axes[2].scatter(c2.best_R2_P1, yy, s=90, color=INK, label="P1", zorder=3)
    axes[2].scatter(c2.best_R2_P2, yy, s=90, facecolors=SURFACE, edgecolors=INK, lw=1.8, label="P2", zorder=3)
    axes[2].set_yticks(yy)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("best R² (Archard / power law)")
    axes[2].legend(loc="lower left")
    fig.subplots_adjust(left=0.22)
    return "C2: wear measures compared (runs measurable, replicate SD, best R² in P1/P2).", "outputs/tables/c2_wear_measures.csv"


# ============================================================================================ 10 physics fits
def _boot_band(kind, df, xg, n_boot, seed):
    from ..evaluate.interpret import group_bootstrap

    P = []
    for idx in group_bootstrap(df.group.to_numpy(), n_boot, seed):
        d = df.iloc[idx]
        P.append(LinearPhysics(kind).fit(d, d.y.to_numpy()).predict(xg)[0])
    return np.percentile(P, [2.5, 97.5], axis=0)


def fig10_physics(D: Data):
    runs = D.runs
    par = D.t("final_parameters")
    perm = D.t("permutation_tests").set_index(["target", "model"])
    met = D.met.set_index(["target", "protocol", "model"])["R2"]
    fig, axes = _fig(2, 2, top=0.86)
    pc = par[(par.target == "COF_first10s")].set_index("param")
    title(fig, f"Load sets the early COF (μ₀ + c·L^(-1/3), c = {pc.at['c', 'estimate']:.3f}) and the track width; volume follows Archard",
          "points = runs (blue training, orange test); line = physics fit on all runs; band = 95 % run-bootstrap interval")
    nb = 300
    for ax, tk in zip(axes.ravel(), ("COF_first10s", "COF_end", "log10_width", "log10_V")):
        t = TARGETS[tk]
        df = target_frame(runs, t)
        m = LinearPhysics(t.physics).fit(df, df.y.to_numpy())
        if tk == "log10_V":
            x = df.log10_LS
            xg = pd.DataFrame({"lnL": np.zeros(50), "log10_LS": np.linspace(x.min() - 0.1, x.max() + 0.1, 50)})
            xs = xg.log10_LS
            ax.set_xlabel("log10 (L·S)  [N·m]")
        else:
            x = df.L
            Lg = np.geomspace(df.L.min() * 0.9, df.L.max() * 1.1, 60)
            xg = pd.DataFrame({"lnL": np.log(Lg), "log10_LS": np.zeros(60)})
            xs = Lg
            ax.set_xscale("log")
            ax.set_xticks([10, 20, 50, 100])
            ax.set_xticklabels(["10", "20", "50", "100"])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.set_xlabel("load L (N)")
        lo, hi = _boot_band(t.physics, df, xg, nb, D.cfg["seed"])
        ax.fill_between(xs, lo, hi, color="#e7e5df", lw=0)
        ax.plot(xs, m.predict(xg)[0], color=INK, lw=2)
        for role, (c, mk) in S.ROLE_STYLE.items():
            dd = df[df.role == role]
            xx = dd.log10_LS if tk == "log10_V" else dd.L
            ax.scatter(xx, dd.y, s=55, color=c, marker=mk, edgecolors=SURFACE, lw=0.8, zorder=3, label=role)
        p = perm["p_value"].get((tk, "physics"), np.nan)
        ax.set_title(f"{t.short}: R² P1 {met[(tk, 'P1', 'physics')]:+.2f}, P2 {met[(tk, 'P2', 'physics')]:+.2f}"
                     + (f", permutation p = {p:.4f}" if np.isfinite(p) else ""), fontsize=12)
        ax.set_ylabel(t.short)
    axes[0, 0].legend(loc="upper right")
    return ("Physics fits: COF_first10s and COF_end vs load (μ₀ + c·L^(-1/3)), track width vs load (log-log), wear volume vs L·S "
            "(Archard, slope 1), each with a 95 % run-bootstrap band."), "outputs/data/runs.csv, outputs/tables/final_parameters.csv"


# ============================================================================================ 11 C3 dot plot
def _clip(v, lo=-1.0):
    return np.maximum(v, lo)


def fig11_c3(D: Data):
    met = D.met
    models = [m for m in MODEL_ORDER if m in set(met.model)]
    fig, axes = plt.subplots(1, 5, figsize=S.SLIDE, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.1, left=0.13, right=0.99, wspace=0.12)
    all_phys = all(D.final(t) in ("physics", "constant") for t in TARGETS)
    title(fig, ("No model beats the physics law robustly" if all_phys else "An ML model beats the physics law on some targets")
          + "; flexible ML models gain in P1 and lose it in P2",
          "R² per model and target: filled = P1 (all runs), hollow = P2 (training runs); grey line = constant, "
          "black line = physics; below -1 drawn at -1")
    yy = np.arange(len(models))[::-1]
    for ax, tk in zip(axes, TARGETS):
        m1 = met[(met.target == tk) & (met.protocol == "P1")].set_index("model")["R2"]
        m2 = met[(met.target == tk) & (met.protocol == "P2")].set_index("model")["R2"]
        ax.axvline(m1.get("constant", np.nan), color=MUTED, lw=1.4)
        ax.axvline(m1.get("physics", np.nan), color=INK, lw=1.4)
        for y, mk in zip(yy, models):
            c = S.MODEL_STYLE[mk][0]
            ax.plot([_clip(m1[mk]), _clip(m2[mk])], [y, y], color=S.GRID, lw=2, zorder=1)
            ax.scatter(_clip(m1[mk]), y, s=75, color=c, marker=S.MODEL_STYLE[mk][1], zorder=3, edgecolors=SURFACE)
            ax.scatter(_clip(m2[mk]), y, s=75, facecolors=SURFACE, edgecolors=c, marker=S.MODEL_STYLE[mk][1], lw=1.6, zorder=3)
        fm = D.final(tk)
        ax.set_title(f"{TLABEL[tk]}\nfinal: {'physics' if fm in ('physics', 'constant') and tk != 'log10_k' else ('Archard k' if tk == 'log10_k' else MODELS[fm].label)}",
                     fontsize=12)
        ax.set_xlim(-1.08, 1.0)
        ax.set_xlabel("R²")
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([MODELS[m].label for m in models])
    return "C3: P1 (filled) and P2 (hollow) R² for every model and target, with constant and physics reference lines.", "outputs/tables/metrics_all.csv"


# ============================================================================================ 12 stability
def fig12_stability(D: Data):
    met = D.met
    p1 = met[met.protocol == "P1"].set_index(["target", "model"])
    p2 = met[met.protocol == "P2"].set_index(["target", "model"])
    fig, ax = _fig(top=0.84)
    fig.subplots_adjust(right=0.78)
    title(fig, "Unstable models win in P1 but fall below the constant in P2",
          "each point = one model x target; colour = model (filled = hybrid / physics), shape = target; axes clipped at -1")
    ax.axhline(0, color=AXISC, lw=1)
    ax.axvline(0, color=AXISC, lw=1)
    ax.plot([-1, 1], [-1, 1], color=AXISC, lw=1)
    ax.fill_between([0, 1], -1.05, 0, color="#fbeee7", lw=0)
    ax.text(0.98, -0.97, "better than zero in P1,\nworse in P2: unstable", ha="right", va="bottom", fontsize=12, color=INK2)
    for (tk, mk), r in p1.iterrows():
        if (tk, mk) not in p2.index:
            continue
        c, _, filled = S.MODEL_STYLE[mk]
        ax.scatter(_clip(r.R2, -1.0), _clip(p2.at[(tk, mk), "R2"], -1.0), s=80, marker=S.TARGET_MARKER[tk],
                   facecolors=c if filled else SURFACE, edgecolors=c if not filled else SURFACE, lw=1.6, zorder=3)
    ax.set_xlim(-1.05, 1)
    ax.set_ylim(-1.05, 1)
    ax.set_xlabel("R² in P1 (all runs)")
    ax.set_ylabel("R² in P2 (training runs only)")
    from matplotlib.lines import Line2D

    h1 = [Line2D([], [], ls="", marker=S.MODEL_STYLE[m][1], markersize=9, markerfacecolor=S.MODEL_STYLE[m][0] if S.MODEL_STYLE[m][2] else SURFACE,
                 markeredgecolor=S.MODEL_STYLE[m][0], label=MODELS[m].label) for m in MODEL_ORDER if m in set(p1.index.get_level_values(1))]
    h2 = [Line2D([], [], ls="", marker=S.TARGET_MARKER[t], markersize=9, color=INK2, label=TLABEL[t]) for t in TARGETS]
    l1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="model", fontsize=11)
    ax.add_artist(l1)
    ax.legend(handles=h2, loc="lower left", bbox_to_anchor=(1.01, 0.0), title="target", fontsize=11)
    return "Stability: P1 vs P2 R² for every model x target; points in the shaded quadrant are unstable.", "outputs/tables/metrics_all.csv"


# ============================================================================================ 13 parity
def fig13_parity(D: Data):
    fig, axes = _fig(1, 5, top=0.8)
    fig.subplots_adjust(wspace=0.35, left=0.05)
    title(fig, "Final models (out-of-fold, P1): the physics laws, with 95 % prediction intervals",
          "measured vs predicted for every run when its replicate group was left out; grey band = ± 2 replicate SD")
    nf = D.t("noise_floor").set_index("target")["replicate_sd"]
    for ax, tk in zip(axes, TARGETS):
        fm = D.final(tk)
        o = D.oof[(D.oof.target == tk) & (D.oof.protocol == "P1") & (D.oof.model == fm)]
        lo, hi = min(o.y.min(), o.pred.min()), max(o.y.max(), o.pred.max())
        pad = 0.08 * (hi - lo)
        g = np.array([lo - pad, hi + pad])
        ax.fill_between(g, g - 2 * nf[tk], g + 2 * nf[tk], color="#efeee9", lw=0)
        ax.plot(g, g, color=AXISC, lw=1)
        c = S.MODEL_STYLE[fm][0]
        ax.errorbar(o.pred, o.y, xerr=1.96 * o.sd, fmt="none", ecolor=S.GRID, elinewidth=1.4, zorder=2)
        for role, (rc, mk) in S.ROLE_STYLE.items():
            d = o[o.role == role]
            ax.scatter(d.pred, d.y, s=45, marker=mk, color=c if role == "training" else rc, edgecolors=SURFACE, zorder=3)
        ax.set_xlim(g)
        ax.set_ylim(g)
        _r2txt(ax, o.y, o.pred)
        cov = np.mean(np.abs(o.pred - o.y) <= 1.96 * o.sd) * 100
        ax.text(0.03, 0.84, f"coverage {cov:.0f} %", transform=ax.transAxes, fontsize=11, color=INK2)
        ax.set_title(f"{TLABEL[tk]}\n{model_label(fm, TARGETS[tk])}", fontsize=11)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("measured")
    return "Parity plots (P1 out-of-fold) of the final model per target with 95 % prediction intervals.", "outputs/data/oof_predictions.csv"


# ============================================================================================ 14 residuals
def fig14_residuals(D: Data):
    fig, axes = plt.subplots(3, 5, figsize=S.SLIDE)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.06, right=0.99, wspace=0.3, hspace=0.45)
    title(fig, "Residuals of the final models show no structure left in load, frequency or time",
          "P1 out-of-fold residual (measured - predicted) vs each input; grey band = ± 2 replicate SD")
    nf = D.t("noise_floor").set_index("target")["replicate_sd"]
    for j, tk in enumerate(TARGETS):
        fm = D.final(tk)
        o = D.oof[(D.oof.target == tk) & (D.oof.protocol == "P1") & (D.oof.model == fm)]
        res = o.y - o.pred
        for i, (col, lab) in enumerate((("L", "load (N)"), ("lnf", "frequency (Hz)"), ("lnt", "time (s)"))):
            ax = axes[i, j]
            x = o.L if col == "L" else np.exp(o[col])
            ax.axhspan(-2 * nf[tk], 2 * nf[tk], color="#efeee9", lw=0)
            ax.axhline(0, color=AXISC, lw=1)
            ax.scatter(x, res, s=28, color=S.MODEL_STYLE[fm][0], edgecolors=SURFACE, lw=0.5, zorder=3)
            if col == "L":
                ax.set_xscale("log")
            ax.tick_params(labelsize=10)
            ax.set_xlabel(lab, fontsize=11)
            if i == 0:
                ax.set_title(TLABEL[tk], fontsize=12)
            if j == 0:
                ax.set_ylabel("residual", fontsize=11)
    return "Residuals of the final models vs load, frequency and time (P1 out-of-fold).", "outputs/data/oof_predictions.csv"


# ============================================================================================ 15 confusion
def fig15_confusion(D: Data):
    c4o = D.d("c4_oof")
    o = D.oof[D.oof.protocol == "P1"]
    cases = [
        ("COF final", o[(o.target == "COF_first10s") & (o.model == D.final("COF_first10s"))], "COF first 10 s, physics"),
        ("COF old pipeline", c4o[(c4o.target == "old_COF") & (c4o.protocol == "P1")], "COF end, ANN"),
        ("COF best plain ANN", o[(o.target == "COF_first10s") & (o.model == "ann")], "COF first 10 s, ANN"),
        ("wear final", o[(o.target == "log10_V") & (o.model == D.final("log10_V"))], "log10 V, Archard"),
        ("wear old pipeline", c4o[(c4o.target == "old_wear") & (c4o.protocol == "P1")], "log10 k (mean loc., plain), ANN"),
        ("wear best plain ANN", o[(o.target == "log10_V") & (o.model == "ann")], "log10 V, ANN"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=S.SLIDE)
    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.97, wspace=0.35, hspace=0.5)
    title(fig, "Tercile confusion: can a model tell low, mid and high runs apart?",
          "rows = measured tercile, columns = predicted tercile (bins from the measured terciles); κ = Cohen's kappa (P1)")
    for ax, (name, d, sub) in zip(axes.ravel(), cases):
        cm, acc, kap = tercile_confusion(d.y.to_numpy(), d.pred.to_numpy())
        ax.imshow(cm, cmap=S.SEQUENTIAL, vmin=0, vmax=max(cm.max(), 1))
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                        color="white" if cm[i, j] > 0.6 * cm.max() else INK)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_yticklabels(["low", "mid", "high"])
        ax.set_title(f"{name}\n{sub}: acc {acc:.2f}, κ {kap:+.2f}", fontsize=11)
        ax.set_xlabel("predicted", fontsize=11)
        ax.set_ylabel("measured", fontsize=11)
    return "Tercile confusion matrices (P1): final vs old pipeline vs best plain ANN, for COF and wear.", "outputs/data/oof_predictions.csv, outputs/data/c4_oof.csv"


# ============================================================================================ 16 uncertainty
def fig16_uncertainty(D: Data):
    met = D.met[(D.met.protocol == "P1") & D.met.coverage95_pct.notna()]
    nf = D.t("noise_floor").set_index("target")["spread_sd"]
    models = [m for m in MODEL_ORDER if m in set(met.model)]
    fig, axes = _fig(1, 2, top=0.84)
    fig.subplots_adjust(left=0.14)
    med = met.groupby("model").coverage95_pct.median()
    ols = med.reindex(["constant", "physics", "powerlaw"]).median()
    title(fig, f"OLS intervals cover {ols:.0f} % (nominal 95 %), GPR {med.get('gpr', np.nan):.0f} %, "
               f"the ANN-ensemble spread only {med.get('ann', np.nan):.0f} %",
          "P1 coverage of the nominal 95 % interval (left) and mean interval width relative to the spread of the target (right)")
    yy = np.arange(len(models))[::-1]
    for y, mk in zip(yy, models):
        d = met[met.model == mk]
        for _, r in d.iterrows():
            kw = S.model_scatter_kw(mk, 70)
            kw["marker"] = S.TARGET_MARKER[r.target]
            axes[0].scatter(r.coverage95_pct, y, zorder=3, **kw)
            axes[1].scatter(r.interval_width / nf[r.target], y, zorder=3, **kw)
    axes[0].axvline(95, color=INK, lw=1.4)
    axes[0].set_xlabel("coverage of the 95 % interval (%)")
    axes[1].set_xlabel("mean interval width / SD of the target over runs")
    for ax in axes:
        ax.set_yticks(yy)
    axes[0].set_yticklabels([MODELS[m].label for m in models])
    axes[1].set_yticklabels([])
    from matplotlib.lines import Line2D

    axes[1].legend(handles=[Line2D([], [], ls="", marker=S.TARGET_MARKER[t], color=INK2, markersize=8, label=TLABEL[t])
                            for t in TARGETS], loc="lower right", fontsize=11)
    return "Uncertainty: 95 % interval coverage and relative width per model (P1).", "outputs/tables/metrics_all.csv"


# ============================================================================================ 17 interpretation
def fig17_interpret(D: Data):
    ls = D.t("gpr_length_scales")
    imp = D.t("ann_permutation_importance")
    fig, axes = _fig(1, 2, top=0.84)
    tks = list(TARGETS)
    g0 = ls[ls.model == "gpr"].set_index("target").reindex(tks)
    n_ls = int((g0[["lnL", "lnf", "lnt"]].idxmin(axis=1) == "lnL").sum())
    a0 = imp[imp.model == "ann"]
    n_imp = int((a0.loc[a0.groupby("target").mse_increase.idxmax()].input == "lnL").sum())
    title(fig, f"Load is the strongest input: shortest GPR length scale in {n_ls} of {len(tks)} targets, "
               f"largest ANN importance in {n_imp} of {len(tks)}",
          "raw GPR and raw ANN fitted on all runs of each target; shorter length scale = stronger effect (capped at 100)")
    x = np.arange(len(tks))
    w = 0.26
    g = ls[ls.model == "gpr"].set_index("target")
    for i, inp in enumerate(("lnL", "lnf", "lnt")):
        v = np.minimum(g.reindex(tks)[inp].to_numpy(), 100)
        axes[0].bar(x + (i - 1) * w, v, width=w - 0.03, color=S.INPUT_COLOR[inp], label=S.INPUT_LABEL[inp])
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TLABEL[t] for t in tks], fontsize=11)
    axes[0].set_ylabel("GPR length scale (standardised inputs)")
    axes[0].set_title("GPR length scales")
    axes[0].legend(loc="upper left")
    a = imp[imp.model == "ann"]
    for i, inp in enumerate(("lnL", "lnf", "lnt")):
        var = {t: target_frame(D.runs, TARGETS[t]).y.var(ddof=0) for t in tks}
        vals = [a[(a.target == t) & (a.input == inp)].mse_increase.iloc[0] / var[t] for t in tks]
        axes[1].bar(x + (i - 1) * w, vals, width=w - 0.03, color=S.INPUT_COLOR[inp], label=S.INPUT_LABEL[inp])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([TLABEL[t] for t in tks], fontsize=11)
    axes[1].set_ylabel("MSE increase when permuted / variance of target")
    axes[1].set_title("ANN permutation importance (group-permuted)")
    return "Interpretation: GPR length scales and ANN permutation importance per input and target.", "outputs/tables/gpr_length_scales.csv, ann_permutation_importance.csv"


# ============================================================================================ 18 surfaces
def fig18_surfaces(D: Data):
    from functools import partial

    from ..evaluate.interpret import fit_on_all
    from ..models.registry import make_model
    from ..stages import most_common_best

    runs = D.runs
    fig, axes = plt.subplots(2, 2, figsize=S.SLIDE)
    fig.subplots_adjust(top=0.84, bottom=0.09, left=0.07, right=0.93, wspace=0.42, hspace=0.42)
    title(fig, "Prediction surfaces over load x frequency (mid run time): contours run almost parallel to the frequency axis",
          "selected ML model per target (physics + ML residual) fitted on all runs; right = predictive SD; dots = runs")
    for row, tk in enumerate(("COF_first10s", "log10_width")):
        t = TARGETS[tk]
        mk = D.selected_ml(tk) or "gpr_phys"
        df = target_frame(runs, t)
        fixed = most_common_best(D.oof, tk, mk) if mk.startswith("ann") else None
        fac = partial(make_model, mk, D.cfg, t, D.cfg["seed"], fixed) if fixed else partial(make_model, mk, D.cfg, t, D.cfg["seed"])
        m = fit_on_all(fac, df)
        L = np.geomspace(df.L.min(), df.L.max(), 60)
        f = np.geomspace(np.exp(df.lnf.min()), np.exp(df.lnf.max()), 50)
        LL, FF = np.meshgrid(L, f)
        tmid = np.exp(np.median(df.lnt))
        g = pd.DataFrame({"lnL": np.log(LL.ravel()), "lnf": np.log(FF.ravel()), "lnt": np.log(tmid), "L": LL.ravel(),
                          "log10_LS": np.log10(LL.ravel() * 0.02 * FF.ravel() * tmid), "group": "grid"})
        mean, sd = m.predict(g)
        for col, (v, lab, cmap) in enumerate(((mean, t.short, S.SEQUENTIAL), (sd, f"SD of {t.short}", S.SEQUENTIAL))):
            ax = axes[row, col]
            cs = ax.contourf(LL, FF, np.asarray(v).reshape(LL.shape), levels=12, cmap=cmap)
            ax.grid(False)
            cb = fig.colorbar(cs, ax=ax, fraction=0.05, pad=0.02, format="%.3g")
            cb.set_label(lab, fontsize=11)
            ax.scatter(df.L, np.exp(df.lnf), s=25, color=INK, edgecolors=SURFACE, lw=0.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([10, 20, 50, 100])
            ax.set_yticks([4, 6, 8, 10, 15])
            _plain_log(ax)
            ax.set_xlabel("load L (N)")
            ax.set_ylabel("frequency f (Hz)")
            ax.set_title(f"{MODELS[mk].label}: {'mean' if col == 0 else 'uncertainty'} ({t.short}, t = {tmid:.0f} s)", fontsize=12)
    return ("Prediction surfaces (mean and SD) of the selected COF and width models over L x f at the median run time.",
            "outputs/data/runs.csv (models refitted on all runs)")


# ============================================================================================ 19 C4
def fig19_c4(D: Data):
    c4 = D.t("c4_old_vs_final")
    c4o = D.d("c4_oof")
    fig = plt.figure(figsize=S.SLIDE)
    oc = c4[(c4.pipeline == "old") & (c4.quantity == "COF")].iloc[0]
    fc = c4[(c4.pipeline == "final") & (c4.quantity == "COF")].iloc[0]
    ow = c4[(c4.pipeline == "old") & (c4.quantity == "wear")].iloc[0]
    fw = c4[(c4.pipeline == "final") & (c4.quantity == "wear")].iloc[0]
    title(fig, f"Old vs final: old COF R² falls to {oc.R2_P2:+.2f} on the training runs (final {fc.R2_P2:+.2f}); "
               f"old wear R² {ow.R2:+.2f} (final {fw.R2:+.2f})",
          "old = plain ANN on end-of-run COF and on mean-of-locations k (plain levelling); final = selected final models; parity = P1")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], left=0.06, right=0.98, top=0.85, bottom=0.09, wspace=0.32, hspace=0.45)
    panels = [("old_COF", "old: COF end, ANN", 0, 0), ("final_COF", "final: COF first 10 s", 0, 1),
              ("old_wear", "old: log10 k (mean loc.), ANN", 1, 0), ("final_wear", "final: log10 V (most-worn)", 1, 1)]
    for key, lab, r, c in panels:
        ax = fig.add_subplot(gs[r, c])
        d = c4o[(c4o.target == key) & (c4o.protocol == "P1")]
        mk = "ann" if key.startswith("old") else d.model.iloc[0]
        lo, hi = min(d.y.min(), d.pred.min()), max(d.y.max(), d.pred.max())
        g = [lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)]
        ax.plot(g, g, color=AXISC, lw=1)
        ax.scatter(d.pred, d.y, s=45, **{k: v for k, v in S.model_scatter_kw(mk, 45).items() if k != "s"})
        ax.set_xlim(g)
        ax.set_ylim(g)
        _r2txt(ax, d.y, d.pred)
        ax.set_title(lab, fontsize=12)
        ax.set_xlabel("predicted", fontsize=11)
        ax.set_ylabel("measured", fontsize=11)
    ax = fig.add_subplot(gs[:, 2])
    ax.set_axis_off()
    cols = ["pipeline", "quantity", "n", "R2", "R2_P2", "tercile_kappa", "replicate_sd", "signal_to_noise"]
    tab = c4[cols].copy()
    for c in ("R2", "R2_P2", "tercile_kappa"):
        tab[c] = tab[c].map(lambda v: f"{v:+.2f}")
    for c in ("replicate_sd",):
        tab[c] = tab[c].map(lambda v: f"{v:.2g}")
    tab["signal_to_noise"] = tab["signal_to_noise"].map(lambda v: f"{v:.1f}")
    tb = ax.table(cellText=tab.values, colLabels=["pipeline", "qty", "n", "R² P1", "R² P2", "κ", "rep. SD", "S/N"],
                  loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(12)
    tb.auto_set_column_width(list(range(len(cols))))
    tb.scale(1, 2.0)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(S.GRID)
        if i == 0:
            cell.set_text_props(fontweight="bold")
    return "C4: old pipeline vs final, metric table and P1 parity plots side by side.", "outputs/tables/c4_old_vs_final.csv, outputs/data/c4_oof.csv"


# ============================================================================================ 20 C5
def fig20_c5(D: Data):
    runs = D.runs
    paper = D.d("paper_table")
    sc = D.t("c5_strategies")
    tr = D.t("c5_trends").set_index(["study", "quantity", "variable"])
    fig = plt.figure(figsize=S.SLIDE)
    best = sc.groupby(["target", "uses_paper"]).R2_P1.max().unstack()
    helps = bool((best[True] > best[False]).any())
    title(fig, "The literature trends disagree with ours, and " + (
        "pooling helps on at least one target" if helps else "no pooling strategy beats our data alone"),
          "values relative to each study's geometric mean (black = ours, red = Ponnuswamy 2024); "
          "exponents from ln y = a + b ln L + c ln v with 95 % bootstrap CI")
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], left=0.06, right=0.985, top=0.84, bottom=0.09, wspace=0.75, hspace=0.62)
    A = D.cfg["experiment"]["stroke_amplitude_mm"]
    det = runs[runs.wear_status == "detected"]
    sets = {
        ("COF", "load"): ((runs.Load_N, runs.COF_first10s), (paper.Load_N, paper.COF)),
        ("COF", "speed"): ((4 * A / 1000 * runs.Freq_Hz, runs.COF_first10s), (paper.Speed_m_s, paper.COF)),
        ("k", "load"): ((det.Load_N, det.k_mm3_Nm), (paper.Load_N, paper.k_paper)),
        ("k", "speed"): ((4 * A / 1000 * det.Freq_Hz, det.k_mm3_Nm), (paper.Speed_m_s, paper.k_paper)),
    }
    for (q, v), ((xo, yo), (xp, yp)) in sets.items():
        ax = fig.add_subplot(gs[0 if q == "COF" else 1, 0 if v == "load" else 1])
        for (x, y), study in (((xo, yo), "ours"), ((xp, yp), "paper")):
            rel = y / np.exp(np.mean(np.log(y)))
            ax.scatter(x, rel, s=35, color=S.STUDY_COLOR[study], edgecolors=SURFACE, lw=0.5, zorder=3)
        eo, ep = tr.loc[("ours", q, v)], tr.loc[("paper", q, v)]
        ax.set_title(f"{q} vs {v}\nours {eo.exponent:+.2f} [{eo.ci_lo:+.2f}, {eo.ci_hi:+.2f}]\n"
                     f"paper {ep.exponent:+.2f} [{ep.ci_lo:+.2f}, {ep.ci_hi:+.2f}]", fontsize=10.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _plain_log(ax)
        ax.set_xlabel("load (N)" if v == "load" else "mean sliding speed (m/s)", fontsize=11)
        ax.set_ylabel(f"relative {q}", fontsize=11)
    ax = fig.add_subplot(gs[:, 2])
    tg = list(dict.fromkeys(sc.target))
    strat = list(sc[sc.target == tg[0]].strategy)
    yy = np.arange(len(strat))[::-1]
    for k, (t, mk) in enumerate(zip(tg, ("o", "^"))):
        d = sc[sc.target == t].set_index("strategy").reindex(strat)
        col = [INK if not u else S.RED for u in d.uses_paper]
        ax.scatter(np.maximum(d.R2_P1, -2), yy + (0.15 if k == 0 else -0.15), c=col, marker=mk, s=60, zorder=3,
                   label=t.replace("log10_", "log10 "))
    ax.axvline(0, color=AXISC, lw=1)
    ax.set_xlim(-2.2, 0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels(strat, fontsize=10)
    ax.set_xlabel("R² on our runs (P1); < -2 drawn at -2", fontsize=11)
    ax.set_title("strategies (black = ours only,\nred = uses the paper rows)", fontsize=11)
    ax.legend(loc="lower left", fontsize=10)
    return ("C5: relative COF and k vs load and speed, ours vs Ponnuswamy 2024, and the P1 scores of the pooling strategies.",
            "inputs/ponnuswamy2024_table5.csv, outputs/tables/c5_trends.csv, c5_strategies.csv")


# ============================================================================================ 21 C6
def fig21_c6(D: Data):
    from ..compare.c6_ta import load_ta_module

    ta = load_ta_module(D.paths.inputs / "analyze_wear.py")
    s, hm = _scan(D, "R24", "m")
    z, px, _ = ta.read_bcrf(str(s.bcrf))
    zo = ta.orient_across_track(z)
    flipped = zo.shape != z.shape
    zl, mask = ta.robust_level(ta.inpaint_voids(zo), order=1, shoulder_fraction=0.15)
    m1, p1, b1 = ta.compute_groove_metrics(zl, px, mask, 10.0)
    zl2, mask2 = ta.robust_level(ta.inpaint_voids(z), order=1, shoulder_fraction=0.15)
    m2, p2, b2 = ta.compute_groove_metrics(zl2, px, mask2, 10.0)
    summ = D.t("c6_ta_summary").set_index("pipeline")
    runs6 = D.t("c6_ta_runs")
    scans6 = D.t("c6_ta_scans")
    fig, axes = _fig(1, 3, top=0.8)
    fig.subplots_adjust(wspace=0.32, bottom=0.22)
    title(fig, f"The TA script reads {int(scans6.flipped.sum())} of {len(scans6)} scans along the track and spreads the replicates "
               f"{summ.at['original TA script', 'replicate_spread_factor']:.0f}x",
          "left: R24.bcrf profile as the TA code reads it vs the correct axis; middle: k of the identical runs R04/R14/R20; right: COF ratio")
    ax = axes[0]
    ax.plot(np.arange(len(p1)) * px, p1 - b1, color=S.RED, lw=1.5,
            label=f"TA ({'transposed' if flipped else 'as read'}): depth {m1['depth_max_um']:.1f} µm, area {m1['area_groove_um2']:.0f} µm²")
    ax.plot(np.arange(len(p2)) * px, p2 - b2, color=INK, lw=1.5,
            label=f"correct axis: depth {m2['depth_max_um']:.1f} µm, area {m2['area_groove_um2']:.0f} µm²")
    ax.axhline(0, color=AXISC, lw=1)
    ax.set_xlabel("profile position (µm)")
    ax.set_ylabel("height vs baseline (µm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=10)
    ax.set_title("R24.bcrf: the 'groove' is a pit", fontsize=12)
    ax = axes[1]
    pipes = ("original TA script", "fixed TA copy", "ours")
    allk = np.array([summ.at[p, f"k_{r}"] for p in pipes for r in ("R04", "R14", "R20")], float)
    ymin = np.nanmin(allk[allk > 0]) / 3
    for i, p in enumerate(pipes):
        for j, r in enumerate(("R04", "R14", "R20")):
            k = summ.at[p, f"k_{r}"]
            if np.isfinite(k) and k > 0:
                ax.scatter(i, k, s=80, color=S.PIPELINE_COLOR[p], marker="osD"[j], edgecolors=SURFACE, zorder=3)
                ax.annotate(r, (i, k), xytext=(8, -12 if r == "R04" else 0), textcoords="offset points", fontsize=10, color=INK2)
            else:
                ax.annotate(f"{r}:\nnot\ndetected", (i, ymin), xytext=(0, 4), textcoords="offset points", fontsize=9,
                            color=INK2, ha="center")
    ax.set_yscale("log")
    ax.set_ylim(ymin * 0.8, np.nanmax(allk) * 2)
    ax.text(0.02, 0.98, f"spread {summ.at['original TA script', 'replicate_spread_factor']:.0f}x / "
            f"{summ.at['fixed TA copy', 'replicate_spread_factor']:.1f}x / {summ.at['ours', 'replicate_spread_factor']:.1f}x",
            transform=ax.transAxes, va="top", fontsize=11, color=INK)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["original", "fixed copy", "ours"])
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel("k of the replicates (mm³/(N·m))")
    ax.set_title("same test, three pipelines", fontsize=12)
    ax = axes[2]
    from scipy import stats

    r, _ = stats.pearsonr(runs6.Freq_Hz, runs6.cof_ratio_orig_to_ours)
    ax.scatter(runs6.Freq_Hz, runs6.cof_ratio_orig_to_ours, s=55, color=S.RED, edgecolors=SURFACE, zorder=3)
    b = np.polyfit(runs6.Freq_Hz, runs6.cof_ratio_orig_to_ours, 1)
    fx = np.linspace(runs6.Freq_Hz.min(), runs6.Freq_Hz.max(), 10)
    ax.plot(fx, np.polyval(b, fx), color=INK, lw=1.5)
    ax.axhline(1, color=AXISC, lw=1)
    ax.set_xlabel("frequency f (Hz)")
    ax.set_ylabel("TA COF / offset-free COF")
    ax.set_title(f"TA COF reads low, more at high f (r = {r:+.2f})", fontsize=12)
    return "C6: TA script - R24 orientation flip, replicate spread original/fixed/ours, COF ratio vs frequency.", \
        "inputs/analyze_wear.py on testdata/, outputs/tables/c6_ta_*.csv"


# ============================================================================================ 22 P3
def fig22_p3(D: Data):
    c4o = D.d("c4_oof")
    fig, axes = _fig(2, 2, top=0.84)
    title(fig, "Hold-out check (P3): train on the 18 training runs, predict the 5 test runs once",
          "these runs were inspected during development, so this is a check, not a sealed test; bars = 95 % intervals where available")
    cases = [("final_COF", "final: COF first 10 s"), ("old_COF", "old: COF end (ANN)"),
             ("final_wear", "final: log10 V (most-worn)"), ("old_wear", "old: log10 k mean loc. (ANN)")]
    for ax, (key, lab) in zip(axes.ravel(), cases):
        d = c4o[(c4o.target == key) & (c4o.protocol == "P3")].sort_values("run_id")
        if d.empty:
            ax.set_axis_off()
            continue
        x = np.arange(len(d))
        mk = "ann" if key.startswith("old") else d.model.iloc[0]
        c = S.MODEL_STYLE[mk][0]
        ax.scatter(x - 0.08, d.y, s=110, facecolors=SURFACE, edgecolors=MUTED, lw=2.2, marker="o", zorder=3, label="measured")
        if d.sd.notna().all():
            ax.errorbar(x + 0.08, d.pred, yerr=1.96 * d.sd, fmt="none", ecolor=S.GRID, elinewidth=2, zorder=2)
        ax.scatter(x + 0.08, d.pred, s=80, zorder=3, label="predicted", **{k: v for k, v in S.model_scatter_kw(mk, 80).items() if k != "s"})
        ax.set_xticks(x)
        ax.set_xticklabels(d.run_id)
        rm = np.sqrt(np.mean((d.pred - d.y) ** 2))
        ax.set_title(f"{lab}: RMSE {rm:.3g}", fontsize=12)
        ax.legend(loc="best", fontsize=10)
    return "P3 hold-out: measured vs predicted for the 5 test runs, final vs old pipeline.", "outputs/data/c4_oof.csv"


FIGURES = [
    ("fig01_pipeline", fig01_pipeline), ("fig02_test_matrix", fig02_test_matrix), ("fig03_raw_signal", fig03_raw_signal),
    ("fig04_cof_time", fig04_cof_time), ("fig05_height_map", fig05_height_map),
    ("fig06_robust_levelling", fig06_robust), ("fig07_wear_per_run", fig07_wear_per_run),
    ("fig08_c1_cof_definitions", fig08_c1), ("fig09_c2_wear_measures", fig09_c2), ("fig10_physics_fits", fig10_physics),
    ("fig11_c3_models", fig11_c3), ("fig12_stability", fig12_stability), ("fig13_parity", fig13_parity),
    ("fig14_residuals", fig14_residuals), ("fig15_confusion", fig15_confusion), ("fig16_uncertainty", fig16_uncertainty),
    ("fig17_interpretation", fig17_interpret), ("fig18_surfaces", fig18_surfaces),
    ("fig19_c4_old_vs_final", fig19_c4), ("fig20_c5_literature", fig20_c5), ("fig21_c6_ta_script", fig21_c6),
    ("fig22_p3_holdout", fig22_p3),
]


def make_all_figures(cfg: dict, paths: Paths, only: list[int] | None = None) -> list[dict]:
    S.apply_style()
    D = Data(cfg, paths)
    out = paths.figures
    index = []
    for i, (name, fn) in enumerate(FIGURES, start=1):
        if only and i not in only:
            continue
        try:
            cap, src = fn(D)
            save(plt.gcf(), out, name)
            index.append({"n": i, "file": f"{name}.png", "caption": cap, "source": src})
            LOG.info("figure %s", name)
        except Exception:
            plt.close("all")
            LOG.error("figure %s failed:\n%s", name, traceback.format_exc())
            index.append({"n": i, "file": f"{name}.png", "caption": "FAILED - see outputs/logs/run.log", "source": ""})
    if not only:
        lines = ["# Figure index", "", "Figures are numbered in slide order. Each exists as PNG (300 dpi) and SVG.", "",
                 "| # | file | caption | data source |", "|---|---|---|---|"]
        lines += [f"| {r['n']} | [{r['file']}]({r['file']}) | {r['caption']} | {r['source']} |" for r in index]
        (out / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index
