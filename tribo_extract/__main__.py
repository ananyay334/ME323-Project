"""Command-line interface.

    python -m tribo_extract organize <dump_dir>      # flat dump -> data/raw/<RunID>/ (+ _unassigned.csv)
    python -m tribo_extract discover data/raw/R02    # Part 0: roles, locations, checkpoints, RAW format
    python -m tribo_extract extract                  # every run folder in data/raw
    python -m tribo_extract extract --run R02 R03    # only these runs (merged into the tables)
    python -m tribo_extract inspect path/to/file.csv|.png|.raw [--cbar-min 45.74 --cbar-max 52.04]
    python -m tribo_extract init-runs                # make empty data/raw/<ID>/ folders from the run sheet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .pipeline import included, load_run_sheet, process_all


def _cmd_extract(a):
    cfg = load_config(a.config)
    process_all(cfg, only=a.run, make_plots=not a.no_plots)


def _cmd_organize(a):
    from .layout import organize
    cfg = load_config(a.config)
    organize(Path(a.dump_dir), cfg)


def _cmd_discover(a):
    """Part-0 helper: classify a run folder's files and resolve every height file's format."""
    from .height_io import read_height
    from .layout import classify_run
    cfg = load_config(a.config)
    d = Path(a.folder)
    sheet = load_run_sheet(cfg["paths"]["run_sheet"]).set_index("Experiment_ID")
    t_run = float(sheet.at[d.name, "Run_time_s"]) if d.name in sheet.index else np.nan
    by_role, scans, issues = classify_run(d, cfg["layout"], t_run)
    for role, files in by_role.items():
        for p in files:
            print(f"{role:>9}  {p.relative_to(d)}  ({p.stat().st_size} bytes)")
    if len(scans):
        print("\n" + scans.assign(height=scans["height"].map(lambda p: p and p.name),
                                  preview=scans["preview"].map(lambda p: p and p.name),
                                  metadata=scans["metadata"].map(lambda p: p and p.name))
              .drop(columns=["dir"]).to_string(index=False))
    for i in issues:
        print("issue:", i)
    for p in by_role["height"]:
        try:
            hm = read_height(p, cfg)
            print(f"\n{p.name}: {hm.z.shape[1]} x {hm.z.shape[0]} px, {hm.dx_um:.3f} um/px, "
                  f"z {np.nanmin(hm.z):.3f}..{np.nanmax(hm.z):.3f} um | {json.dumps(hm.info, default=str)}")
        except Exception as exc:                                     # noqa: BLE001
            print(f"\n{p.name}: {exc}")


def _cmd_inspect(a):
    cfg = load_config(a.config)
    p = Path(a.file)
    out_dir = Path(cfg["paths"]["reports_dir"]).parent / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv" and "_height" not in p.name.lower():
        from .cof import extract_cof, extract_cof_cycles
        from .plots import plot_cof, plot_cof_cycles
        from .tribometer_csv import read_tribometer_csv
        run = read_tribometer_csv(p, cfg["csv"]["columns"])
        cps, cycles, cint = extract_cof_cycles(run, cfg, load_N=a.load, freq_hz=a.freq, stroke_mm=a.stroke)
        print(json.dumps({"meta": run.meta, "columns": run.column_map, "flags": cint["flags"],
                          "checkpoints": cps}, indent=2, default=float))
        if len(cycles):
            plot_cof_cycles(cycles, cps, p.stem, out_dir / f"{p.stem}_cof_cycles.png")
        else:                                                        # e.g. the 100 Hz practice run
            res, ts, oint = extract_cof(run, cfg, load_N=a.load)
            plot_cof(oint, res, p.stem, out_dir / f"{p.stem}_cof.png")
    else:
        from .height_io import read_height
        from .plots import plot_wear
        from .wear import measure_wear_scar
        from .wli import read_wli_screenshot
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            hm = read_wli_screenshot(p, cfg, cbar_min=a.cbar_min, cbar_max=a.cbar_max)
        else:
            hm = read_height(p, cfg)
        res, wint = measure_wear_scar(hm, cfg, angle_deg=a.angle)
        print(json.dumps({"z_um": [hm.zmin, hm.zmax], "objective": hm.objective,
                          "px_um": [hm.dx_um, hm.dy_um], "masked_frac": hm.masked_frac,
                          "info": hm.info, **res}, indent=2, default=float))
        if res["wear_detected"]:
            print(f"groove volume = {res['worn_area_um2'] * 1e-6 * cfg['wear']['track_length_mm']:.4e} mm^3 "
                  f"(area x {cfg['wear']['track_length_mm']} mm)")
        plot_wear(hm, wint, res, p.name, out_dir / f"{p.stem}_wear.png")
    print(f"\nfigure written to {out_dir}")


def _cmd_init(a):
    cfg = load_config(a.config)
    sheet = included(load_run_sheet(cfg["paths"]["run_sheet"]))
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

    o = sub.add_parser("organize", help="copy a flat dump into data/raw/<RunID>/")
    o.add_argument("dump_dir")
    o.set_defaults(func=_cmd_organize)

    d = sub.add_parser("discover", help="Part 0: file roles, locations, checkpoints, RAW format of a run folder")
    d.add_argument("folder")
    d.set_defaults(func=_cmd_discover)

    e = sub.add_parser("extract", help="process run folders into data/processed tables")
    e.add_argument("--run", nargs="*", help="Experiment_IDs to (re)process; default all")
    e.add_argument("--no-plots", action="store_true")
    e.set_defaults(func=_cmd_extract)

    i = sub.add_parser("inspect", help="analyse a single CSV, PNG or height file without the run sheet")
    i.add_argument("file")
    i.add_argument("--cbar-min", type=float, default=None)
    i.add_argument("--cbar-max", type=float, default=None)
    i.add_argument("--angle", type=float, default=None, help="track angle, deg CCW from +x")
    i.add_argument("--load", type=float, default=None)
    i.add_argument("--freq", type=float, default=None, help="nominal frequency (fallback cycle count)")
    i.add_argument("--stroke", type=float, default=None, help="stroke amplitude in mm (default config)")
    i.set_defaults(func=_cmd_inspect)

    n = sub.add_parser("init-runs", help="create empty data/raw/<ID>/ folders for every run-sheet row")
    n.set_defaults(func=_cmd_init)

    a = ap.parse_args(argv)
    np.seterr(all="ignore")
    a.func(a)


if __name__ == "__main__":
    main()
