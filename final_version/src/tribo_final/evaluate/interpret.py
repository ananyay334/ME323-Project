"""Interpretation: run-bootstrap parameter intervals, power-law exponents, GPR length scales, ANN permutation
importance, physical-consistency check, and k as one material constant."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..models.baselines import PowerLaw
from ..models.base import INPUTS
from ..models.physics import HERTZ_WIDTH_B, LinearPhysics


def group_bootstrap(groups: np.ndarray, n_boot: int, seed: int):
    """Yield row-index arrays resampling replicate groups with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ug}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        yield np.concatenate([members[g] for g in pick])


def physics_params_ci(kind: str, df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    m = LinearPhysics(kind).fit(df, df["y"].to_numpy())
    est = m.params
    boots = {k: [] for k in est}
    for idx in group_bootstrap(df["group"].to_numpy(), n_boot, seed):
        d = df.iloc[idx]
        try:
            p = LinearPhysics(kind).fit(d, d["y"].to_numpy()).params
        except np.linalg.LinAlgError:
            continue
        for k in est:
            boots[k].append(p[k])
    rows = []
    for k, v in est.items():
        b = np.array(boots[k])
        rows.append({"param": k, "estimate": v, "ci_lo": np.percentile(b, 2.5), "ci_hi": np.percentile(b, 97.5),
                     "boot_sd": b.std(ddof=1)})
    out = pd.DataFrame(rows)
    if kind == "width":  # Hertz expects b = (1/3)/ln10; also report the exponent w ~ L^beta
        r = out[out.param == "b"].iloc[0]
        f = np.log(10)
        out = pd.concat([out, pd.DataFrame([{"param": "exponent_beta (w ~ L^beta)", "estimate": r.estimate * f,
                                             "ci_lo": r.ci_lo * f, "ci_hi": r.ci_hi * f, "boot_sd": r.boot_sd * f},
                                            {"param": "Hertz expectation for b", "estimate": HERTZ_WIDTH_B,
                                             "ci_lo": np.nan, "ci_hi": np.nan, "boot_sd": np.nan}])])
    return out


def power_law_exponents(df: pd.DataFrame, log10_target: bool, n_boot: int, seed: int,
                        inputs: list[str] | None = None) -> pd.DataFrame:
    """d ln y / d ln x for each input (OLS of ln y on the ln inputs), with run-bootstrap 95 % intervals."""
    inputs = inputs or INPUTS
    lny = df["y"].to_numpy() * np.log(10) if log10_target else np.log(df["y"].to_numpy())
    d = df.assign(y=lny)

    def fit(dd):
        return PowerLaw(inputs).fit(dd, dd["y"].to_numpy()).coef

    est = fit(d)
    boots = {k: [] for k in inputs}
    for idx in group_bootstrap(d["group"].to_numpy(), n_boot, seed):
        c = fit(d.iloc[idx])
        for k in inputs:
            boots[k].append(c[k])
    return pd.DataFrame([{"input": k, "exponent": est[k], "ci_lo": np.percentile(boots[k], 2.5),
                          "ci_hi": np.percentile(boots[k], 97.5)} for k in inputs])


def gpr_length_scales(model) -> dict:
    gp = model.residual if hasattr(model, "residual") else model
    return gp.length_scales


def ann_importance(model, df: pd.DataFrame, n_repeats: int, seed: int) -> pd.DataFrame:
    """Group-permuted importance on the fitted model: each input's values are shuffled between replicate groups
    (replicates move together) and the increase in MSE is recorded."""
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    base = float(np.mean((model.predict(df)[0] - y) ** 2))
    groups = df["group"].to_numpy()
    ug = np.unique(groups)
    rows = []
    for col in INPUTS:
        vals = {g: df.loc[groups == g, col].iloc[0] for g in ug}
        inc = []
        for _ in range(n_repeats):
            perm = dict(zip(ug, rng.permutation(ug)))
            d = df.copy()
            d[col] = [vals[perm[g]] for g in groups]
            inc.append(float(np.mean((model.predict(d)[0] - y) ** 2)) - base)
        rows.append({"input": col, "mse_increase": np.mean(inc), "mse_increase_sd": np.std(inc, ddof=1),
                     "relative_increase": np.mean(inc) / base if base > 0 else np.nan})
    return pd.DataFrame(rows)


def grid_frame(df: pd.DataFrame, A_mm: float, n: int = 5) -> pd.DataFrame:
    """L x f x t grid over the observed ranges (log-spaced), with the physics columns."""
    lnL = np.linspace(df.lnL.min(), df.lnL.max(), n)
    lnf = np.linspace(df.lnf.min(), df.lnf.max(), n)
    lnt = np.linspace(df.lnt.min(), df.lnt.max(), n)
    G = np.array(np.meshgrid(lnL, lnf, lnt, indexing="ij")).reshape(3, -1).T
    g = pd.DataFrame(G, columns=INPUTS)
    g["L"] = np.exp(g.lnL)
    S = 4 * A_mm / 1000 * np.exp(g.lnf) * np.exp(g.lnt)  # nominal cycles f*t
    g["log10_LS"] = np.log10(g.L * S)
    g["group"] = "grid"
    return g


def physical_consistency(model, physics_kind: str, df: pd.DataFrame, rep_sd: float, A_mm: float) -> dict:
    """'Never far from the physics law': on an L x f x t grid over the observed ranges, the 90th percentile of
    |model - physics| (both fitted on all runs of the target) must stay within 2 replicate SD."""
    phys = LinearPhysics(physics_kind).fit(df, df["y"].to_numpy())
    g = grid_frame(df, A_mm)
    dev = np.abs(model.predict(g)[0] - phys.predict(g)[0])
    p90 = float(np.percentile(dev, 90))
    return {"max_dev_from_physics": float(dev.max()), "p90_dev_from_physics": p90,
            "physically_consistent": bool(p90 <= 2 * rep_sd) if np.isfinite(rep_sd) else np.nan}


def k_material_constant(k: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    lk = np.log10(k)
    gm = 10 ** lk.mean()
    gsd = 10 ** lk.std(ddof=1)
    boots = [10 ** lk[idx].mean() for idx in group_bootstrap(groups, n_boot, seed)]
    return {"k_geo_mean": gm, "k_gsd": gsd, "k_ci_lo": np.percentile(boots, 2.5), "k_ci_hi": np.percentile(boots, 97.5),
            "n": len(k), "log10_k_mean": lk.mean(), "log10_k_sd": lk.std(ddof=1)}


def noise_floor(df: pd.DataFrame) -> dict:
    """Replicate SD from groups with >= 2 runs (pooled), and spread over runs / replicate SD."""
    num, dof, members = 0.0, 0, []
    for g, d in df.groupby("group"):
        if len(d) >= 2:
            num += float(np.sum((d.y - d.y.mean()) ** 2))
            dof += len(d) - 1
            members += list(d.run_id)
    rep_sd = np.sqrt(num / dof) if dof else np.nan
    spread = float(df.y.std(ddof=1))
    return {"replicate_sd": rep_sd, "replicate_runs": ",".join(members), "replicate_dof": dof,
            "spread_sd": spread, "signal_to_noise": spread / rep_sd if rep_sd and np.isfinite(rep_sd) else np.nan,
            "n_runs": len(df)}


def fit_on_all(factory: Callable, df: pd.DataFrame):
    return factory().fit(df, df["y"].to_numpy(), df["group"].to_numpy())
