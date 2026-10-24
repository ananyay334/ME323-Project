"""Top-up run selection: where the models are least sure (GPR σ + ANN ensemble spread).

    python -m ml.topup --n 4 [--data real|synthetic] [--config ...] [--quick]

* candidates: 15 x 12 x 7 log-spaced points over L in [10, 100] N, f in [4, 15] Hz,
  t in [60, 120] s, t rounded to an even number of seconds (two equal steps), kept only if
  t <= 120 (f <= 10) or t <= 120 - 12 (f - 10) (f > 10), and 7200 <= L f t <= 90000;
* score = z(GPR latent σ) + z(ANN ensemble SD), averaged over log10 k and COF_steady at the
  100 % checkpoint (t_ckpt = t), both models fitted on all training rows;
* greedy picks, each >= ``min_distance`` from every existing run and earlier pick in
  standardised log space (ln L, ln f, ln t standardised by the training runs);
* free positions: F1/F2 x X {8, 21, 34} x Y {3 .. 39}, minus the positions in the run sheet;
* written to data/topup_candidates.csv (synthetic: data/synthetic/topup_candidates.csv) with
  the ANN ensemble's predictions ± SD at both checkpoints: the pre-registered predictions.
At most ``topup.max_runs`` (10) top-up runs in total.
"""
from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from .config import data_paths, load_config, resolve_path, set_global_seeds, setup_logging
from .data import INPUTS, load_data
from .evaluate import fit_full, predict_target
from .models.registry import get_spec
from .splits import SplitPlan

log = logging.getLogger("ml.topup")


def satisfies(L, f, t, cfg: dict) -> np.ndarray:
    """Element-wise DOE constraints of a candidate (L, f, t)."""
    L, f, t = (np.asarray(a, float) for a in (L, f, t))
    tc = cfg["topup"]
    tmax = np.where(f <= 10, 120.0, 120.0 - 12.0 * (f - 10.0))
    lft = L * f * t
    return ((t <= tmax + 1e-9) & (lft >= tc["min_lft"]) & (lft <= tc["max_lft"]) & (np.mod(t, 2) == 0)
            & (L >= 10 - 1e-9) & (L <= 100 + 1e-9) & (f >= 4 - 1e-9) & (f <= 15 + 1e-9)
            & (t >= 60) & (t <= 120))


def candidate_grid(cfg: dict) -> pd.DataFrame:
    g = cfg["topup"]["grid"]
    L = np.round(np.geomspace(g["L"][0], g["L"][1], int(g["L"][2])), 2)            # settable values
    f = np.round(np.geomspace(g["f"][0], g["f"][1], int(g["f"][2])), 2)
    t = 2 * np.round(np.geomspace(g["t"][0], g["t"][1], int(g["t"][2])) / 2)        # even seconds
    LL, FF, TT = np.meshgrid(L, f, t, indexing="ij")
    c = pd.DataFrame({"Load_N": LL.ravel(), "Freq_Hz": FF.ravel(), "Run_time_s": TT.ravel()})
    c = c[satisfies(c["Load_N"], c["Freq_Hz"], c["Run_time_s"], cfg)]
    return c.drop_duplicates().reset_index(drop=True)


def zscore(a: np.ndarray) -> np.ndarray:
    s = np.std(a)
    return (a - np.mean(a)) / s if s > 0 else np.zeros_like(a)


def greedy_pick(Z: np.ndarray, score: np.ndarray, existing: np.ndarray, n: int, min_dist: float) -> list[int]:
    """Highest score first, each pick >= min_dist (Euclidean) from existing points and earlier picks."""
    picks: list[int] = []
    pts = [p for p in existing]
    for i in np.argsort(-score, kind="stable"):
        if len(picks) >= n:
            break
        if all(np.linalg.norm(Z[i] - p) >= min_dist for p in pts):
            picks.append(int(i))
            pts.append(Z[i])
    return picks


def free_positions(sheet: pd.DataFrame, cfg: dict) -> list[tuple[str, int, int]]:
    tc = cfg["topup"]
    used = set()
    for r in sheet.itertuples(index=False):
        face, x, y = getattr(r, "Face", None), getattr(r, "X_mm", None), getattr(r, "Y_mm", None)
        if isinstance(face, str) and pd.notna(x) and pd.notna(y):
            used.add((face.strip(), int(round(float(x))), int(round(float(y)))))
    return [(fc, int(x), int(y)) for fc in tc["faces"] for x in tc["x_mm"] for y in tc["y_mm"]
            if (fc, int(x), int(y)) not in used]


def select_topups(cfg: dict, source: str, n: int, log=log.info) -> pd.DataFrame:
    data = load_data(cfg, source, split="train")
    sheet = pd.read_csv(resolve_path(cfg["data"]["run_sheet"]), dtype={"Experiment_ID": str, "Face": str})
    role = sheet.get("Role", pd.Series("", index=sheet.index)).astype(str).str.lower()
    existing_ids = sheet.loc[role == "topup", "Experiment_ID"].tolist()
    room = int(cfg["topup"]["max_runs"]) - len(existing_ids)
    if n > room:
        log(f"only {max(room, 0)} more top-up run(s) allowed (max {cfg['topup']['max_runs']}); n = {n} capped")
        n = max(room, 0)
    cand = candidate_grid(cfg)
    Xc = pd.DataFrame({"ln_L": np.log(cand["Load_N"]), "ln_f": np.log(cand["Freq_Hz"]),
                       "ln_t": np.log(cand["Run_time_s"])})
    plan = SplitPlan(data.groups, cfg)
    gpr, ann = get_spec("gpr"), get_spec("ann")
    ann_est, _ = fit_full(ann, data, cfg, plan)
    score = np.zeros(len(cand))
    for t in data.targets:
        g_est, _ = fit_full(gpr, data, cfg, plan, t)
        cand[f"gpr_sigma_{t}"] = g_est.predict_latent_std(Xc)
        cand[f"ann_sd_{t}"] = predict_target(ann_est, ann, Xc, data.targets, t)[1]
        score += zscore(cand[f"gpr_sigma_{t}"].to_numpy()) + zscore(cand[f"ann_sd_{t}"].to_numpy())
    cand["score"] = score / len(data.targets)
    # standardised log space of the training runs (100 % checkpoint: t = t_run)
    runs = data.frame[data.frame["Checkpoint"] == data.frame.groupby("Experiment_ID")["Checkpoint"].transform("max")]
    ref = runs[list(INPUTS)].to_numpy(float)
    mu, sd = ref.mean(0), ref.std(0)
    Z = (Xc.to_numpy(float) - mu) / sd
    sh = sheet.dropna(subset=["Load_N", "Freq_Hz", "Run_time_s"])
    ex = (np.log(sh[["Load_N", "Freq_Hz", "Run_time_s"]].to_numpy(float)) - mu) / sd
    picks = greedy_pick(Z, cand["score"].to_numpy(), ex, n, float(cfg["topup"]["min_distance"]))
    pos = free_positions(sheet, cfg)
    if len(pos) < len(picks):
        raise SystemExit(f"only {len(pos)} free positions for {len(picks)} top-up runs")
    k0 = max([int(m.group(1)) for e in existing_ids if (m := re.match(r"T(\d+)", str(e)))] + [0])
    rows = []
    for k, i in enumerate(picks):
        c = cand.iloc[i]
        face, x, y = pos[k]
        row = {"Experiment_ID": f"T{k0 + k + 1:02d}", "Load_N": float(c["Load_N"]),
               "Freq_Hz": float(c["Freq_Hz"]), "Run_time_s": int(c["Run_time_s"]),
               "Face": face, "X_mm": x, "Y_mm": y}
        for j in (1, 2):
            Xp = pd.DataFrame({"ln_L": [np.log(row["Load_N"])], "ln_f": [np.log(row["Freq_Hz"])],
                               "ln_t": [np.log(row["Run_time_s"] * j / 2)]})
            for t in data.targets:
                m, s = predict_target(ann_est, ann, Xp, data.targets, t)
                row[f"{t}_cp{j}"], row[f"{t}_cp{j}_sd"] = float(m[0]), float(s[0])
        row.update({k2: float(c[k2]) for k2 in c.index if k2.startswith(("gpr_sigma_", "ann_sd_")) or k2 == "score"})
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="select top-up runs (GPR sigma + ANN ensemble spread)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--data", choices=["real", "synthetic"], default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="first grid value, ANN ensemble of 3")
    ap.add_argument("--out", default=None, help="CSV path (default: data.<source>.topup_candidates)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, quick=a.quick)
    setup_logging()
    set_global_seeds(cfg["seed"])
    source = a.data or cfg["data"]["source"]
    out = resolve_path(a.out) if a.out else data_paths(cfg, source)["topup_candidates"]
    df = select_topups(cfg, source, a.n)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("%d top-up run(s) -> %s\n%s", len(df), out,
             df[["Experiment_ID", "Load_N", "Freq_Hz", "Run_time_s", "Face", "X_mm", "Y_mm", "score"]]
             .to_string(index=False) if len(df) else "")
    return df


if __name__ == "__main__":
    main()
