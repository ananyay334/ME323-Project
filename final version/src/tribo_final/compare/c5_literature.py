"""C5 - Does the literature data help?

Ponnuswamy & Manoharan 2024 (Mater. Res. Express 11 066515), Table 5: 18 pin-on-disc runs at 10-20 N, 1-3 m/s,
200 s. k_paper = V / (L v 200 s); our mean sliding speed v = 4 A f.

Trends: power-law exponents of COF and k vs load and speed, ours vs paper, with bootstrap CIs.
Strategies, scored ONLY on our runs under P1 (LOGO), with the paper rows always in the training set:
ours alone, naive pooling (power law / GPR / ANN), pooled with a study offset, our power law shrunk toward the
paper's slopes (strength by inner CV), paper slopes imposed, GPR / ANN with a study flag.

The published COF (0.002-0.007) and wear volumes (up to 1620 mm^3 in 200 s, about the whole pin) are physically
implausible for dry steel sliding, so this is mainly a check that pooling cannot help.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from ..evaluate.cv import collect_oof, run_tasks
from ..evaluate.interpret import group_bootstrap
from ..evaluate.metrics import compute_metrics
from ..models.ann import ANNEnsemble
from ..models.base import OLS, Model
from ..models.baselines import PowerLaw
from ..models.gpr import GPR

BASE = ["lnL", "lnf", "lnt"]
FLAG = BASE + ["study"]


class ImposedSlopes(Model):
    """Paper slopes for ln L and ln f fixed; intercept and ln t slope fitted on the training rows of OUR study."""

    name = "Paper slopes imposed"
    has_uncertainty = True

    def __init__(self, slopes: dict):
        self.slopes = slopes

    def _off(self, X):
        return self.slopes["lnL"] * X["lnL"].to_numpy() + self.slopes["lnf"] * X["lnf"].to_numpy()

    def fit(self, X, y, groups=None):
        ours = X["study"].to_numpy() == 0
        Phi = np.column_stack([np.ones(ours.sum()), X["lnt"].to_numpy()[ours]])
        self.ols = OLS().fit(Phi, np.asarray(y)[ours] - self._off(X[ours]))
        return self

    def predict(self, X):
        m, sd = self.ols.predict(np.column_stack([np.ones(len(X)), X["lnt"].to_numpy()]))
        return m + self._off(X), sd


class ShrunkPowerLaw(Model):
    """Our power law with a ridge penalty pulling the ln L and ln f slopes toward the paper's slopes; the penalty
    strength is chosen by inner GroupKFold on our training rows."""

    name = "Shrunk toward paper slopes"
    has_uncertainty = False
    GRID = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0]

    def __init__(self, slopes: dict, inner_folds: int = 5):
        self.slopes = slopes
        self.inner_folds = inner_folds

    @staticmethod
    def _solve(A, y, lam, prior):
        # minimise |y - A b|^2 + lam * sum_j (b_j - prior_j)^2 over the penalised columns (L, f)
        P = np.diag([0.0, 1.0, 1.0, 0.0])
        b0 = np.array([0.0, prior[0], prior[1], 0.0])
        return np.linalg.solve(A.T @ A + lam * P, A.T @ y + lam * P @ b0)

    def fit(self, X, y, groups=None):
        ours = X["study"].to_numpy() == 0
        A = np.column_stack([np.ones(ours.sum()), X.loc[ours, BASE].to_numpy()])
        yy = np.asarray(y)[ours]
        g = np.asarray(groups)[ours] if groups is not None else np.arange(len(yy))
        prior = (self.slopes["lnL"], self.slopes["lnf"])
        k = min(self.inner_folds, len(np.unique(g)))
        best, best_err = 0.0, np.inf
        for lam in self.GRID:
            err = 0.0
            for tr, te in GroupKFold(k).split(A, yy, g):
                b = self._solve(A[tr], yy[tr], lam, prior)
                err += float(np.sum((A[te] @ b - yy[te]) ** 2))
            if err < best_err - 1e-12:
                best, best_err = lam, err
        self.lam = best
        self.coef = self._solve(A, yy, best, prior)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), X[BASE].to_numpy()]) @ self.coef, None


class OursOnly(Model):
    """Wrapper that drops the paper rows before fitting (the 'our data alone' reference)."""

    def __init__(self, inner: Model):
        self.inner = inner
        self.name = inner.name
        self.has_uncertainty = inner.has_uncertainty

    def fit(self, X, y, groups=None):
        ours = X["study"].to_numpy() == 0
        self.inner.fit(X[ours], np.asarray(y)[ours], None if groups is None else np.asarray(groups)[ours])
        return self

    def predict(self, X):
        return self.inner.predict(X)


def _make(kind: str, cfg: dict, slopes: dict):
    seed = cfg["seed"]
    m = cfg["models"]
    return {
        "ours: power law": lambda: OursOnly(PowerLaw(BASE)),
        "ours: GPR": lambda: OursOnly(GPR(m["gpr"], seed, inputs=BASE)),
        "ours: ANN": lambda: OursOnly(ANNEnsemble(m["ann"], seed, inputs=BASE)),
        "naive pooling: power law": lambda: PowerLaw(BASE),
        "naive pooling: GPR": lambda: GPR(m["gpr"], seed, inputs=BASE),
        "naive pooling: ANN": lambda: ANNEnsemble(m["ann"], seed, inputs=BASE),
        "pooled + study offset: power law": lambda: PowerLaw(FLAG),
        "ours shrunk toward paper slopes": lambda: ShrunkPowerLaw(slopes, cfg["evaluate"]["inner_folds"]),
        "paper slopes imposed": lambda: ImposedSlopes(slopes),
        "GPR + study flag": lambda: GPR(m["gpr"], seed, inputs=FLAG),
        "ANN + study flag": lambda: ANNEnsemble(m["ann"], seed, inputs=FLAG),
    }[kind]()


STRATEGIES = ["ours: power law", "ours: GPR", "ours: ANN", "naive pooling: power law", "naive pooling: GPR",
              "naive pooling: ANN", "pooled + study offset: power law", "ours shrunk toward paper slopes",
              "paper slopes imposed", "GPR + study flag", "ANN + study flag"]


def paper_frame(paper: pd.DataFrame, A_mm: float) -> pd.DataFrame:
    p = paper.copy()
    p["k_paper"] = p.Wear_volume_mm3 / (p.Load_N * p.Speed_m_s * p.Time_s)
    p["f_equiv_Hz"] = p.Speed_m_s / (4 * A_mm / 1000)
    return p


def _pooled(runs: pd.DataFrame, paper: pd.DataFrame, target: str, A_mm: float) -> pd.DataFrame:
    if target == "log10_COF":
        o = runs.assign(y=np.log10(runs.COF_first10s))
        p = paper.assign(y=np.log10(paper.COF))
    else:
        o = runs[runs.wear_status == "detected"].assign(y=np.log10(runs.k_mm3_Nm))
        p = paper.assign(y=np.log10(paper.k_paper))
    ours = pd.DataFrame({"run_id": o.run_id, "role": o.role, "group": o.replicate_group, "lnL": np.log(o.Load_N),
                         "lnf": np.log(o.Freq_Hz), "lnt": np.log(o.Run_time_s), "study": 0, "y": o.y,
                         "L": o.Load_N})
    lit = pd.DataFrame({"run_id": p.Run, "role": "paper", "group": p.Run, "lnL": np.log(p.Load_N),
                        "lnf": np.log(p.f_equiv_Hz), "lnt": np.log(p.Time_s), "study": 1, "y": p.y, "L": p.Load_N})
    return pd.concat([ours, lit], ignore_index=True)


def trend_exponents(runs: pd.DataFrame, paper: pd.DataFrame, A_mm: float, n_boot: int, seed: int) -> pd.DataFrame:
    """ln y = a + b ln L + c ln v (+ d ln t for ours): exponents b (load) and c (speed) with bootstrap CIs."""
    rows = []
    det = runs[runs.wear_status == "detected"]
    sets = {
        ("ours", "COF"): (runs, np.log(runs.COF_first10s), runs.replicate_group, True),
        ("paper", "COF"): (paper, np.log(paper.COF), paper.Run, False),
        ("ours", "k"): (det, np.log(det.k_mm3_Nm), det.replicate_group, True),
        ("paper", "k"): (paper, np.log(paper.k_paper), paper.Run, False),
    }
    for (study, q), (d, y, g, has_t) in sets.items():
        if study == "ours":
            X = np.column_stack([np.log(d.Load_N), np.log(4 * A_mm / 1000 * d.Freq_Hz), np.log(d.Run_time_s)])
        else:
            X = np.column_stack([np.log(d.Load_N), np.log(d.Speed_m_s)])
        y = np.asarray(y, float)
        A = np.column_stack([np.ones(len(y)), X])
        est = np.linalg.lstsq(A, y, rcond=None)[0]
        B = np.array([np.linalg.lstsq(A[i], y[i], rcond=None)[0] for i in group_bootstrap(np.asarray(g), n_boot, seed)])
        for j, name in ((1, "load"), (2, "speed")):
            rows.append({"study": study, "quantity": q, "variable": name, "exponent": est[j],
                         "ci_lo": np.percentile(B[:, j], 2.5), "ci_hi": np.percentile(B[:, j], 97.5)})
    return pd.DataFrame(rows)


def paper_slopes(pooled: pd.DataFrame) -> dict:
    p = pooled[pooled.study == 1]
    A = np.column_stack([np.ones(len(p)), p.lnL, p.lnf])
    c = np.linalg.lstsq(A, p.y.to_numpy(), rcond=None)[0]
    return {"lnL": float(c[1]), "lnf": float(c[2])}


def run_c5(cfg: dict, runs: pd.DataFrame, paper_raw: pd.DataFrame, n_jobs: int):
    A_mm = cfg["experiment"]["stroke_amplitude_mm"]
    paper = paper_frame(paper_raw, A_mm)
    trends = trend_exponents(runs, paper, A_mm, cfg["evaluate"]["n_bootstrap"], cfg["seed"])
    tasks = []
    for target in ("log10_COF", "log10_k"):
        df = _pooled(runs, paper, target, A_mm)
        slopes = paper_slopes(df)
        idx = np.arange(len(df))
        ours = idx[df.study.to_numpy() == 0]
        lit = idx[df.study.to_numpy() == 1]
        g = df.group.to_numpy()[ours]
        for s in STRATEGIES:
            fac = partial(_make, s, cfg, slopes)
            for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(ours, groups=g)):
                tasks.append({"target": target, "strategy": s, "fold": fold, "factory": fac, "df": df,
                              "tr": np.r_[ours[tr], lit], "te": ours[te], "cost": 30 if "ANN" in s else 1})
    oof = collect_oof(tasks, run_tasks(tasks, n_jobs), ["target", "strategy"])
    rows = []
    for (target, s), g in oof.groupby(["target", "strategy"], sort=False):
        m = compute_metrics(g.y.to_numpy(), g.pred.to_numpy())
        rows.append({"target": target, "strategy": s, "uses_paper": not s.startswith("ours:"), "n": m["n"],
                     "R2_P1": m["R2"], "RMSE_P1": m["RMSE"], "bias": m["bias"]})
    scores = pd.DataFrame(rows)
    parts = []
    for target, g in scores.groupby("target", sort=False):
        own = g[~g.uses_paper].R2_P1.max()
        pooled = g[g.uses_paper]
        best = pooled.loc[pooled.R2_P1.idxmax()]
        parts.append(f"{target}: best with paper data {best.strategy} R² {best.R2_P1:+.2f} vs ours alone {own:+.2f}")
    tr = trends.set_index(["study", "quantity", "variable"])["exponent"]
    verdict = (f"Do not use the literature data: paper COF ~ L^{tr[('paper', 'COF', 'load')]:+.2f} v^{tr[('paper', 'COF', 'speed')]:+.2f} "
               f"vs ours L^{tr[('ours', 'COF', 'load')]:+.2f} v^{tr[('ours', 'COF', 'speed')]:+.2f}; " + "; ".join(parts)
               + ". The published COF (0.002-0.007) and volumes (up to 1620 mm³ in 200 s) are implausible as printed.")
    return trends, scores, oof, paper, verdict
