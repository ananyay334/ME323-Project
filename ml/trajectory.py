"""Extension B: time-resolved wear trajectory (two-stage model, DOE eq. 4).

    V_w(t) = Vdot_steady * t + dV_run * (1 - exp(-t / tau_trans))

When runs carry the eq.-4 parameters (``Vdot_steady_mm3_per_s, dV_run_mm3,
tau_trans_s``, fitted by the extraction from >= 4 checkpoint scans), each parameter is
modelled (log10, since they are positive and skewed) with nested group CV on the same
fold plan as the main analysis. The out-of-fold parameter predictions are turned back
into curves with :func:`tribo_extract.physics.two_stage_wear` and compared with the
measured checkpoint volumes (``wear_checkpoints`` table when available, otherwise
the measured eq.-4 fit evaluated at the checkpoint times).

Parameters pinned at a bound of the extraction's curve fit are not measurements:
``tau_trans`` at its lower bound (<= ``tau_bound_s``) and ``Vdot_steady`` / ``dV_run``
more than ``floor_decades`` decades below the median (or <= 0). They are excluded from
that parameter's model and counted (QC rule decided before any modelling). With
checkpoints at 0/60/150/300/450/600 s, only one scan falls inside a 20-90 s running-in,
so tau_trans is weakly identified; earlier checkpoints (e.g. 15, 30 s) would help.
If the columns are absent or empty this module does nothing.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tribo_extract.physics import two_stage_wear

from . import plotting as P
from .config import stable_seed
from .data import TARGETS, TriboData
from .evaluate import CVResult, cross_validate, make_design
from .models.registry import select_models
from .splits import FoldPlan

log = logging.getLogger("ml.trajectory")

PARAMS = {"log10_Vdot_steady": "Vdot_steady_mm3_per_s", "log10_dV_run": "dV_run_mm3",
          "log10_tau_trans": "tau_trans_s"}
MIN_RUNS = 10


def trajectory_runs(data: TriboData) -> pd.DataFrame:
    """CV runs with all three eq.-4 parameters finite (empty if the columns are absent)."""
    cols = list(PARAMS.values())
    r = data.cv_runs
    if not set(cols) <= set(r.columns):
        return pd.DataFrame()
    v = r[cols].apply(pd.to_numeric, errors="coerce")
    return r[np.isfinite(v).all(axis=1)]


def at_bound_mask(runs: pd.DataFrame, tcfg: dict) -> pd.DataFrame:
    """Boolean frame (one column per eq.-4 parameter): True where the fit hit a bound."""
    out = pd.DataFrame(index=runs.index)
    dec = float(tcfg.get("floor_decades", 3.0))
    for p, col in PARAMS.items():
        v = pd.to_numeric(runs[col], errors="coerce")
        if p == "log10_tau_trans":
            out[col] = v <= float(tcfg.get("tau_bound_s", 0.01))
        else:
            med = v[v > 0].median()
            out[col] = (v <= 0) | (v < med * 10 ** (-dec))
    return out & runs[list(PARAMS.values())].notna().to_numpy()


def measured_checkpoints(data: TriboData, ids: list[str], times: list[float]) -> pd.DataFrame:
    """Measured V(t) per run: checkpoint table if present, else the measured eq.-4 fit."""
    if data.checkpoints is not None and len(data.checkpoints):
        cp = data.checkpoints[data.checkpoints["Experiment_ID"].isin(ids)].copy()
        cp = cp.rename(columns={"Wear_Volume_mm3": "V_meas"})
        cp["source"] = "checkpoint scans"
        return cp[["Experiment_ID", "t_s", "V_meas", "source"]].dropna(subset=["V_meas"])
    r = data.runs.set_index("Experiment_ID").loc[ids]
    rows = [{"Experiment_ID": e, "t_s": t, "source": "measured eq.-4 fit",
             "V_meas": float(two_stage_wear(t, r.at[e, PARAMS["log10_Vdot_steady"]],
                                            r.at[e, PARAMS["log10_dV_run"]],
                                            r.at[e, PARAMS["log10_tau_trans"]]))}
            for e in ids for t in times]
    return pd.DataFrame(rows)


def reconstruct(cv: CVResult, data: TriboData, times: list[float], n_mc: int = 200,
                seed: int = 0) -> pd.DataFrame:
    """Predicted V(t) at the checkpoint times from out-of-fold parameter predictions (repeat 0)."""
    p = cv.predictions[cv.predictions["repeat"] == 0]
    piv = p.pivot_table(index=["model", "feature_set", "Experiment_ID"], columns="target",
                        values=["y_pred", "y_std"], aggfunc="first")
    rows = []
    rng = np.random.default_rng(seed)
    t = np.asarray(times, float)
    for (m, fs, e), r in piv.iterrows():
        mu = [r.get(("y_pred", k), np.nan) for k in PARAMS]
        if not np.all(np.isfinite(mu)):
            continue
        sd = [r.get(("y_std", k), np.nan) for k in PARAMS]
        v = two_stage_wear(t, *(10 ** np.asarray(mu)))
        lo = hi = np.full(len(t), np.nan)
        if np.all(np.isfinite(sd)):
            draws = 10 ** (np.asarray(mu)[:, None] + np.asarray(sd)[:, None] * rng.standard_normal((3, n_mc)))
            vs = two_stage_wear(t[:, None], draws[0], draws[1], draws[2])
            lo, hi = np.percentile(vs, 2.5, axis=1), np.percentile(vs, 97.5, axis=1)
        rows += [{"model": m, "feature_set": fs, "Experiment_ID": e, "t_s": tt, "V_pred": vv,
                  "V_lo": a, "V_hi": b} for tt, vv, a, b in zip(t, v, lo, hi)]
    return pd.DataFrame(rows)


def curve_metrics(curves: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """Errors of the reconstructed V(t) at the measured checkpoints (t > 0)."""
    m = curves.merge(meas, on=["Experiment_ID", "t_s"])
    m = m[m["t_s"] > 0]
    out = []
    for (mod, fs), g in m.groupby(["model", "feature_set"]):
        y, yh = g["V_meas"].to_numpy(float), g["V_pred"].to_numpy(float)
        ss = float(((y - y.mean()) ** 2).sum())
        last = g[g["t_s"] == g["t_s"].max()]
        inside = (g["V_meas"] >= g["V_lo"]) & (g["V_meas"] <= g["V_hi"])
        out.append({"model": mod, "feature_set": fs, "n_runs": g["Experiment_ID"].nunique(),
                    "n_points": len(g), "RMSE_V_mm3": float(np.sqrt(np.mean((y - yh) ** 2))),
                    "R2_V": 1 - float(((y - yh) ** 2).sum()) / ss if ss > 0 else np.nan,
                    "MdAPE_final_pct": float(100 * np.median(np.abs(last["V_pred"] / last["V_meas"] - 1))),
                    "band95_coverage": float(inside[np.isfinite(g["V_lo"])].mean())
                    if np.isfinite(g["V_lo"]).any() else np.nan})
    return pd.DataFrame(out).sort_values("RMSE_V_mm3").reset_index(drop=True)


@P.styled
def curves_figure(curves: pd.DataFrame, meas: pd.DataFrame, data: TriboData, model: str, fs: str,
                  n: int, out: Path) -> Path:
    c = curves[(curves["model"] == model) & (curves["feature_set"] == fs)]
    ids = sorted(c["Experiment_ID"].unique())
    v_end = meas[meas["t_s"] == meas["t_s"].max()].set_index("Experiment_ID")["V_meas"]
    ids = sorted(ids, key=lambda e: v_end.get(e, 0.0))
    ids = [ids[i] for i in np.unique(np.linspace(0, len(ids) - 1, min(n, len(ids))).round().astype(int))]
    nc = min(3, len(ids))
    nr = int(np.ceil(len(ids) / nc))
    fig, ax = P.new_fig(nr, nc, w=3.5 * nc, h=2.7 * nr)
    tt = np.linspace(0, float(meas["t_s"].max()), 200)
    runs = data.runs.set_index("Experiment_ID")
    col = P.model_color(model)
    for k, e in enumerate(ids):
        a = ax[k // nc, k % nc]
        ce = c[c["Experiment_ID"] == e].sort_values("t_s")
        fit = two_stage_wear(tt, *(runs.at[e, PARAMS[p]] for p in PARAMS))
        a.plot(tt, fit, color=P.AXIS, lw=1.2, label="measured eq.-4 fit")
        if np.isfinite(ce["V_lo"]).any():
            a.fill_between(ce["t_s"], ce["V_lo"], ce["V_hi"], color=col, alpha=0.1, lw=0)
        a.plot(ce["t_s"], ce["V_pred"], color=col, label=f"predicted ({model}, {fs})")
        me = meas[meas["Experiment_ID"] == e]
        P.dots(a, me["t_s"], me["V_meas"], P.INK, size=24, label="measured checkpoints")
        a.set_title(e, fontsize=9)
        a.set_xlabel("time [s]" if k // nc == nr - 1 else "")
        a.set_ylabel("wear volume [mm³]" if k % nc == 0 else "")
        a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    for k in range(len(ids), nr * nc):
        ax[k // nc, k % nc].set_visible(False)
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=len(lab))
    fig.suptitle("Wear trajectory V(t): out-of-fold prediction vs measurement (band = 95 %)",
                 x=0.01, ha="left", color=P.INK, fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return P.save(fig, out)


def run_trajectory(data: TriboData, cfg: dict, plan: FoldPlan, out_dir: Path, n_jobs: int = 1,
                   model_names: list[str] | None = None) -> dict | None:
    """Extension B; returns None (and does nothing) without trajectory data."""
    tcfg = cfg["trajectory"]
    traj = trajectory_runs(data)
    if len(traj) < MIN_RUNS:
        log.info("Extension B skipped: %d CV run(s) with eq.-4 trajectory parameters (< %d)",
                 len(traj), MIN_RUNS)
        return None
    bound = at_bound_mask(data.runs, tcfg)
    n_bound = {c: int(bound.loc[data.runs["Experiment_ID"].isin(traj["Experiment_ID"]), c].sum())
               for c in bound}
    if any(n_bound.values()):
        log.info("Extension B: fit-bound parameter values excluded: %s", n_bound)
    runs = data.runs.copy()
    for c in bound:
        runs.loc[bound[c], c] = np.nan
    data = dataclasses.replace(data, runs=runs)
    names = [m for m in tcfg["models"] if not model_names or m in model_names]
    specs, skipped = select_models(cfg, names) if names else ([], {})
    designs = [d for d in (make_design(data, t, fs, cfg) for t in PARAMS for fs in tcfg["feature_sets"])
               if d is not None]
    info = {t: d.info for t, d in ((d.target, d) for d in designs)}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv = cross_validate(designs, specs, cfg, plan, n_jobs, label="trajectory CV")
    summary = cv.summary()
    summary.to_csv(out_dir / "trajectory_param_metrics.csv", index=False)
    times = [float(t) for t in tcfg.get("checkpoint_times_s", [0, 60, 150, 300, 450, 600])]
    ids = sorted(traj["Experiment_ID"])
    meas = measured_checkpoints(data, ids, times)
    cp_times = sorted(set(times) | set(meas["t_s"].astype(float)))
    curves = reconstruct(cv, data, cp_times, seed=stable_seed(cfg["seed"], "traj"))
    res = {"n_runs": len(traj), "summary": summary, "info": info, "skipped_models": skipped,
           "n_at_bound": n_bound,
           "meas_source": meas["source"].iloc[0] if len(meas) else None, "figures": {}}
    if curves.empty:
        return res
    cm = curve_metrics(curves, meas)
    cm.to_csv(out_dir / "trajectory_curve_metrics.csv", index=False)
    curves.to_csv(out_dir / "trajectory_curves.csv", index=False)
    res["curve_metrics"] = cm
    best = cm.iloc[0]
    res["figures"]["curves"] = curves_figure(
        curves, meas, data, best["model"], best["feature_set"], int(tcfg.get("n_curve_plots", 9)),
        Path(out_dir).parent / "figures" / "trajectory_curves.png")
    res["cv"] = cv
    return res
