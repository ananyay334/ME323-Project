"""Test-day (session) effect check on the TRAINING runs only.

The campaign was run on two days (CSV header ``TestDate``). This asks whether the day explains
COF or track width beyond load / frequency / time, with leave-one-group-out CV of a plain
linear model (the R04/R14/R20 replicates are one group). Test-role runs are never read.

    python -m ml.session_effect            # -> ml/results/session_effect.md (+ .csv)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROLES = {"completed", "corner", "interior", "center_replicate", "topup"}
MODELS = [("intercept only", []), ("day", ["day"]), ("ln L", ["lnL"]), ("ln L + day", ["lnL", "day"]),
          ("ln L, ln f, ln t", ["lnL", "lnf", "lnt"]), ("ln L, ln f, ln t + day", ["lnL", "lnf", "lnt", "day"])]


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["Role"].astype(str).str.lower().isin(TRAIN_ROLES) & (df["Checkpoint"] == 2)].copy()
    if "Test_date" not in df or df["Test_date"].isna().all():
        raise SystemExit("runs_checkpoints.csv has no Test_date: re-run `python -m tribo_extract extract`.")
    day = pd.to_datetime(df["Test_date"], errors="coerce").dt.normalize()
    days = sorted(day.dropna().unique())
    df["day_label"] = day.dt.strftime("%Y-%m-%d")
    df["day"] = (day != days[0]).astype(float)                 # 0 = first test day, 1 = any later day
    df["lnL"], df["lnf"], df["lnt"] = np.log(df["Load_N"]), np.log(df["Freq_Hz"]), np.log(df["Run_time_s"])
    df["log10_w"] = np.log10(df["Track_width_um"])
    rep = df["Replicate_of"].fillna("").astype(str).str.strip()
    df["group"] = np.where(rep.isin(["", "nan"]), df["Experiment_ID"].astype(str), rep)
    return df


def logo(df: pd.DataFrame, target: str, cols: list[str]) -> tuple[float, float, dict]:
    d = df.dropna(subset=[target, *cols])
    y = d[target].to_numpy(float)
    X = d[cols].to_numpy(float) if cols else np.zeros((len(d), 1))
    e = np.empty_like(y)
    for g in d["group"].unique():
        te = (d["group"] == g).to_numpy()
        e[te] = LinearRegression().fit(X[~te], y[~te]).predict(X[te]) - y[te]
    coef = dict(zip(cols, LinearRegression().fit(X, y).coef_)) if cols else {}
    return float(np.sqrt(np.mean(e ** 2))), float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), coef


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", default=str(ROOT / "data" / "processed" / "runs_checkpoints.csv"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "results" / "session_effect"))
    a = ap.parse_args(argv)
    df = load_train(Path(a.table))
    rows, md = [], ["# Test-day effect (training runs only, leave-one-group-out CV)", ""]
    for lab, g in df.groupby("day_label"):
        md.append(f"- {lab}: {', '.join(sorted(g['Experiment_ID']))}")
    md.append("")
    for t in ("COF_steady", "log10_w"):
        m = df.groupby("day")[t].agg(["mean", "std", "count"])
        md += [f"## {t}", "", f"day means: first day {m.loc[0, 'mean']:.4f} (n={int(m.loc[0, 'count'])}), "
               f"later day {m.loc[1, 'mean']:.4f} (n={int(m.loc[1, 'count'])})", "",
               "| model | LOGO RMSE | LOGO R² | coefficients |", "|---|---|---|---|"]
        for name, cols in MODELS:
            rmse, r2, coef = logo(df, t, cols)
            rows.append({"target": t, "model": name, "RMSE": rmse, "R2": r2, **{f"coef_{k}": v for k, v in coef.items()}})
            md.append(f"| {name} | {rmse:.4f} | {r2:+.2f} | " + ", ".join(f"{k} {v:+.4f}" for k, v in coef.items()) + " |")
        md.append("")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out + ".csv", index=False)
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
