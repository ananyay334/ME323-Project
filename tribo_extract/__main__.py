"""Command-line interface.

    python -m tribo_extract extract                 # every run folder in data/raw
    python -m tribo_extract extract --run R001 R002 # only these runs (merged into the tables)
    python -m tribo_extract inspect path/to/file.csv|.png [--cbar-min 45.74 --cbar-max 52.04]
    python -m tribo_extract init-runs               # make empty data/raw/<ID>/ folders from the run sheet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .pipeline import load_run_sheet, process_all


def _cmd_extract(a):
    cfg = load_config(a.config)
    process_all(cfg, only=a.run, make_plots=not a.no_plots)


def _cmd_inspect(a):
    cfg = load_config(a.config)
    p = Path(a.file)
    out_dir = Path(cfg["paths"]["reports_dir"]).parent / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        from .cof import extract_cof
        from .plots import plot_cof
        from .tribometer_csv import read_tribometer_csv
        run = read_tribometer_csv(p, cfg["csv"]["columns"])
        res, ts, cint = extract_cof(run, cfg, load_N=a.load)
        print(json.dumps({"meta": run.meta, "columns": run.column_map, **res}, indent=2, default=float))
        print(ts.head(10).to_string())
        plot_cof(cint, res, p.stem, out_dir / f"{p.stem}_cof.png")
    else:
        from .plots import plot_wear
        from .wear import measure_wear_scar
        from .wli import read_wli_screenshot
        hm = read_wli_screenshot(p, cfg, cbar_min=a.cbar_min, cbar_max=a.cbar_max)
        res, wint = measure_wear_scar(hm, cfg, angle_deg=a.angle)
        print(json.dumps({"cbar_um": [hm.zmin, hm.zmax], "objective": hm.objective,
                          "px_um": [hm.dx_um, hm.dy_um], "masked_frac": hm.masked_frac,
                          "screenshot_info": hm.info, **res}, indent=2, default=float))
        if res["wear_detected"] and a.stroke:
            print(f"wear volume = {res['worn_area_um2'] * 1e-6 * a.stroke:.4e} mm^3 (A x stroke)")
        plot_wear(hm, wint, res, p.name, out_dir / f"{p.stem}_wear.png")
    print(f"\nfigure written to {out_dir}")


def _cmd_init(a):
    cfg = load_config(a.config)
    sheet = load_run_sheet(cfg["paths"]["run_sheet"])
    raw = Path(cfg["paths"]["raw_dir"])
    n = 0
    for eid in sheet["Experiment_ID"]:
        d = raw / eid
        if not d.exists():
            d.mkdir(parents=True)
            n += 1
    print(f"created {n} run folders in {raw}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tribo_extract", description="COF & wear-rate target extraction")
    ap.add_argument("--config", default=None, help="alternative YAML config")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="process run folders into data/processed tables")
    e.add_argument("--run", nargs="*", help="Experiment_IDs to (re)process; default all")
    e.add_argument("--no-plots", action="store_true")
    e.set_defaults(func=_cmd_extract)

    i = sub.add_parser("inspect", help="analyse a single CSV or PNG without the run sheet")
    i.add_argument("file")
    i.add_argument("--cbar-min", type=float, default=None)
    i.add_argument("--cbar-max", type=float, default=None)
    i.add_argument("--angle", type=float, default=None, help="track angle, deg CCW from +x")
    i.add_argument("--load", type=float, default=None)
    i.add_argument("--stroke", type=float, default=None, help="track length in mm (for a volume)")
    i.set_defaults(func=_cmd_inspect)

    n = sub.add_parser("init-runs", help="create empty data/raw/<ID>/ folders for every run-sheet row")
    n.set_defaults(func=_cmd_init)

    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    a.func(a)


if __name__ == "__main__":
    main()
