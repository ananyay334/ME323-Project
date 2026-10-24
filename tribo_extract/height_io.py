"""Read a height scan (RAW binary or text grid) into a :class:`~tribo_extract.wli.HeightMap`.

BCR-F files (``.bcrf`` / ``.bcr``, the RTEC inline profilometer's processed height map) are read
directly from their self-describing header (:func:`read_bcrf`). Other RAW formats are resolved per
file with a ladder, stopping at the first success:

1. format keys: the ``raw:`` block of config/file_layout.yaml, else a companion metadata
   file (txt / xml / ini / json / hdr) giving nx, ny, dtype, header offset, z scale/unit
   and pixel size;
2. a text file -> :func:`~tribo_extract.wli.read_height_matrix`;
3. brute force: header offsets x dtypes x nx*ny factor pairs (aspect 0.25-4) of the
   remaining size, ranked by normalised total variation (a wrong row length shears the
   image and multiplies the row-to-row differences); the top 5 are printed.

Validation: Pearson r > ``preview_min_r`` against the PNG preview of the same scan (decoded
with :func:`~tribo_extract.wli.read_wli_screenshot`, resampled to one shape, all four flips
tried), else a plausible height range after levelling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .wli import HeightMap, read_height_matrix, read_wli_screenshot

META_EXT = (".txt", ".xml", ".ini", ".json", ".hdr", ".inf")
_BRUTE_CACHE: dict = {}
# metadata keys -> regex alternatives (value follows "=", ":" or a tag)
META_KEYS = {
    "nx": r"(?:nx|x[_ ]?(?:size|res(?:olution)?|points|pixels)|width|columns|cols|samples)",
    "ny": r"(?:ny|y[_ ]?(?:size|res(?:olution)?|points|pixels)|height|rows|lines)",
    "header_bytes": r"(?:header[_ ]?(?:size|bytes|offset|length)|offset|data[_ ]?offset)",
    "dtype": r"(?:dtype|data[_ ]?type|format|pixel[_ ]?type|bits(?:[_ ]?per[_ ]?pixel)?)",
    "z_scale": r"(?:z[_ ]?(?:scale|factor|spacing|res(?:olution)?|step))",
    "z_unit": r"(?:z[_ ]?unit|height[_ ]?unit|unit)",
    "pixel_um": r"(?:pixel[_ ]?size|x[_ ]?(?:spacing|step|pitch)|lateral[_ ]?res(?:olution)?|dx)",
}
UNIT_UM = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "micron": 1.0, "microns": 1.0, "nm": 1e-3,
           "a": 1e-4, "å": 1e-4}


def _raw_cfg(cfg: dict) -> dict:
    return (cfg.get("layout") or {}).get("raw") or {}


def companion(path: Path, exts: tuple) -> Path | None:
    """Same-scan companion file (same name up to the first '.') with one of ``exts``."""
    stem = path.name.split(".")[0]
    for q in sorted(path.parent.iterdir()):
        if q != path and q.name.split(".")[0] == stem and q.suffix.lower() in exts:
            return q
    return None


def parse_metadata(path: Path) -> dict:
    """Best-effort key/value scan of a metadata file (txt / ini / xml / json / hdr)."""
    txt = path.read_bytes()[:65536].decode("utf-8", errors="replace")
    flat: dict[str, str] = {}
    if path.suffix.lower() == ".json":
        try:
            def walk(d, pre=""):
                for k, v in (d.items() if isinstance(d, dict) else []):
                    if isinstance(v, dict):
                        walk(v, f"{pre}{k}_")
                    else:
                        flat[f"{pre}{k}".lower()] = str(v)
            walk(json.loads(txt))
        except ValueError:
            pass
    for m in re.finditer(r"<\s*([\w.\- ]+?)\s*>\s*([^<]+?)\s*<\s*/", txt):           # <tag>v</tag>
        flat.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    for m in re.finditer(r'([\w.\- ]+?)\s*=\s*"([^"]*)"', txt):                        # attr="v"
        flat.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    for m in re.finditer(r"^\s*([\w.\- ()µ]+?)\s*[:=\t]\s*(.+?)\s*$", txt, re.M):        # key = v
        flat.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    out: dict = {}
    for key, pat in META_KEYS.items():
        for k, v in flat.items():
            if re.fullmatch(rf"(?:[\w.\- ]*[_ .])?{pat}(?:\s*\(.*\))?", k, re.I):
                out[key] = v
                if key in ("z_scale", "pixel_um"):
                    u = re.search(r"\(([^)]+)\)", k)
                    if u:
                        out[key + "_unit"] = u.group(1).strip().lower()
                break
    res: dict = {"source": str(path)}
    for k in ("nx", "ny", "header_bytes"):
        if k in out and re.search(r"\d+", out[k]):
            res[k] = int(float(re.search(r"[-+]?\d+(?:\.\d+)?", out[k]).group()))
    if "dtype" in out:
        d = out["dtype"].lower()
        res["dtype"] = ("<f8" if re.search(r"double|f8|float64|64", d) else
                        "<f4" if re.search(r"float|real|f4|32f", d) else
                        "<i4" if re.search(r"int32|i4|long", d) else
                        "<u2" if re.search(r"uint16|u2|unsigned", d) else
                        "<i2" if re.search(r"int16|i2|short|16", d) else None)
    for k in ("z_scale", "pixel_um"):
        if k in out:
            m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", out[k])
            if m:
                unit = out.get(k + "_unit") or (re.sub(r"[\d.eE+\- ]", "", out[k]).lower() or None)
                f = UNIT_UM.get(unit, 1.0) if unit else 1.0
                res["z_scale_um" if k == "z_scale" else "pixel_um"] = float(m.group()) * f
    if "z_unit" in out and "z_scale_um" not in res and out["z_unit"].lower() in UNIT_UM:
        res["z_scale_um"] = UNIT_UM[out["z_unit"].lower()]
    return res


def _is_text(path: Path) -> bool:
    head = path.read_bytes()[:4096]
    if not head or b"\x00" in head:
        return False
    try:
        s = head.decode("ascii")
    except UnicodeDecodeError:
        return False
    return bool(re.search(r"\d", s)) and sum(c.isprintable() or c in "\r\n\t" for c in s) > 0.98 * len(s)


def _normalised_tv(z: np.ndarray) -> float:
    """Median |row / column differences| over the 5-95 % value spread (robust to a few
    header or invalid values read as heights)."""
    z = z[: min(len(z), 512)]
    p5, p95 = np.nanpercentile(z, [5, 95])
    if not np.isfinite(p95 - p5) or p95 == p5:
        return np.inf
    return float((np.nanmedian(np.abs(np.diff(z, axis=0))) + np.nanmedian(np.abs(np.diff(z, axis=1))))
                 / (p95 - p5))


def _load(path: Path, dtype: str, offset: int, n: int | None = None) -> np.ndarray:
    a = np.fromfile(path, dtype=np.dtype(dtype), offset=offset, count=-1 if n is None else n)
    a = a.astype(float)
    if np.dtype(dtype).kind in "iu":
        info = np.iinfo(np.dtype(dtype))
        a[(a == info.min) | (a == info.max)] = np.nan                    # invalid-pixel markers
    a[~np.isfinite(a) | (np.abs(a) > 1e12)] = np.nan
    return a


def brute_force(path: Path, rc: dict, verbose: bool = True) -> list[dict]:
    """Ranked (offset, dtype, nx, ny) candidates for a headerless/unknown RAW file."""
    size = path.stat().st_size
    ck = (size, tuple(rc.get("offsets", [])), tuple(rc.get("dtypes", [])))
    if ck in _BRUTE_CACHE:
        return _BRUTE_CACHE[ck]
    lo, hi = rc.get("aspect", [0.25, 4.0])
    cands = []
    for off in rc.get("offsets", [0, 128, 256, 512, 1024, 2048, 4096]):
        for dt in rc.get("dtypes", ["<f4", "<f8", "<i2", "<u2", "<i4"]):
            item = np.dtype(dt).itemsize
            if size <= off or (size - off) % item:
                continue
            n = (size - off) // item
            a = _load(path, dt, off)
            if np.isfinite(a).mean() < 0.5 or np.nanstd(a) == 0:
                continue
            for ny in range(int(np.sqrt(n / hi)), int(np.sqrt(n / lo)) + 2):
                if ny < 8 or n % ny:
                    continue
                nx = n // ny
                if not lo <= nx / ny <= hi:
                    continue
                cands.append({"offset": off, "dtype": dt, "nx": nx, "ny": ny,
                              "tv": _normalised_tv(a.reshape(ny, nx))})
    cands.sort(key=lambda c: c["tv"])
    if verbose:
        print(f"[height_io] brute force on {path.name} ({size} bytes): top 5 of {len(cands)}")
        for c in cands[:5]:
            print(f"    offset {c['offset']:>5}  dtype {c['dtype']}  nx {c['nx']:>5}  ny {c['ny']:>5}  "
                  f"norm. TV {c['tv']:.4f}")
    _BRUTE_CACHE[ck] = cands
    return cands


def _guess_z_scale(z: np.ndarray, rng_um) -> tuple[float, bool]:
    """Unit factor that puts the levelled height range in ``rng_um`` (um, nm, mm, m tried)."""
    from .wear import level_plane
    zl = level_plane(z, 1)
    span = float(np.nanpercentile(zl, 99.5) - np.nanpercentile(zl, 0.5))
    for f in (1.0, 1e-3, 1e3, 1e6):
        if rng_um[0] <= span * f <= rng_um[1]:
            return f, True
    return 1.0, False


def _resample(a: np.ndarray, shape) -> np.ndarray:
    filled = np.where(np.isfinite(a), a, np.nanmean(a))
    return ndi.zoom(filled, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)


def validate_against_preview(hm: HeightMap, preview: Path, cfg: dict) -> dict:
    """Pearson r between the levelled RAW map and the decoded PNG preview (best flip)."""
    from .wear import level_plane
    try:
        pv = read_wli_screenshot(preview, cfg)
    except Exception as exc:                                             # noqa: BLE001
        return {"preview_r": np.nan, "preview_error": f"{type(exc).__name__}: {exc}"[:120]}
    a = level_plane(pv.z, 1)
    b = _resample(level_plane(hm.z, 1), a.shape)
    ok = np.isfinite(a)
    best = (-np.inf, None)
    for name, f in (("none", lambda x: x), ("flipud", np.flipud), ("fliplr", np.fliplr),
                    ("rot180", lambda x: x[::-1, ::-1])):
        r = float(np.corrcoef(a[ok], f(b)[ok])[0, 1])
        best = max(best, (r, name), key=lambda t: t[0] if np.isfinite(t[0]) else -np.inf)
    return {"preview_r": best[0], "preview_flip": best[1], "preview_file": preview.name}


def read_bcrf(path: str | Path) -> HeightMap:
    """BCR-F height map (SPIP / RTEC export): text header + nx*ny values, z in um.

    Handles the unicode (UTF-16, header = 2 x ``headersize`` bytes) and ASCII variants,
    ``bcrf`` float32 data and ``bcr`` int16 data (x ``bit2nm``). ``voidpixels`` -> NaN.
    Lengths are in ``xunit``/``yunit`` (nm or um), heights in ``zunit`` (um or nm).
    """
    path = Path(path)
    raw = path.read_bytes()
    unicode = raw[1:2] == b"\x00"
    text = raw[:4096].decode("utf-16-le", errors="replace") if unicode else raw[:2048].decode("latin-1", "replace")
    hdr = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            hdr[k.strip().lower()] = v.strip().strip("\x00")
    hsize = int(float(hdr.get("headersize", 2048))) * (2 if unicode else 1)
    nx, ny = int(hdr["xpixels"]), int(hdr["ypixels"])
    fmt = hdr.get("fileformat", "bcrf").lower()
    little = hdr.get("intelmode", "1") == "1"
    dt = np.dtype(("<" if little else ">") + ("f4" if fmt.startswith("bcrf") else "i2"))
    z = np.frombuffer(raw, dtype=dt, count=nx * ny, offset=hsize).reshape(ny, nx).astype(float)
    void = float(hdr.get("voidpixels", "3.402823E+38"))
    z[(z >= 0.999 * void) | ~np.isfinite(z)] = np.nan
    if not fmt.startswith("bcrf"):
        z = z * float(hdr.get("bit2nm", 1)) / 1000.0
    if hdr.get("zunit", "um").lower() == "nm":
        z = z / 1000.0

    def um(val, unit):
        return float(val) / 1000.0 if unit.lower() == "nm" else float(val)
    dx = um(hdr["xlength"], hdr.get("xunit", "nm")) / nx
    dy = um(hdr["ylength"], hdr.get("yunit", "nm")) / ny
    info = {"read_method": "bcrf", "nx": nx, "ny": ny, "dtype": str(dt), "offset": hsize,
            "z_scale_source": "bcrf_header", "pixel_source": "bcrf_header",
            "stage_offset_um": [um(hdr.get("xoffset", "nan"), hdr.get("xunit", "nm")),
                                um(hdr.get("yoffset", "nan"), hdr.get("yunit", "nm"))]}
    from .wear import level_plane
    span = float(np.subtract(*np.nanpercentile(level_plane(z, 1), [99.5, 0.5])))
    info["levelled_range_um"] = span
    info["validated"] = bool(np.isfinite(span) and 0.1 <= span <= 500.0)
    return HeightMap(z=z, dx_um=dx, dy_um=dy, source=str(path), zmin=float(np.nanmin(z)),
                     zmax=float(np.nanmax(z)), masked_frac=float(np.isnan(z).mean()), info=info)


def read_height(path: str | Path, cfg: dict, verbose: bool = True) -> HeightMap:
    """Height map in um (NaN = invalid) with lateral pixel size; ``info`` records the method."""
    path = Path(path)
    if path.suffix.lower() in (".bcrf", ".bcr"):
        return read_bcrf(path)
    rc = _raw_cfg(cfg)
    fov = cfg["wli"]["fov_um"][rc.get("objective") or cfg["wli"]["default_objective"]]
    info: dict = {}
    if _is_text(path):
        hm = read_height_matrix(path, fov)
        hm.info.update({"read_method": "text_grid", "pixel_source": "fov_fallback"})
        if rc.get("pixel_um"):
            hm.dx_um = hm.dy_um = float(rc["pixel_um"])
            hm.info["pixel_source"] = "file_layout"
        return hm

    meta_file = companion(path, META_EXT)
    meta = parse_metadata(meta_file) if meta_file else {}
    fmt = {k: rc.get(k) for k in ("nx", "ny", "dtype", "header_bytes", "z_scale_um", "pixel_um")
           if rc.get(k) is not None}
    method = "file_layout" if {"nx", "ny", "dtype"} <= set(fmt) else None
    if method is None and {"nx", "ny", "dtype"} <= set(meta):
        fmt = {**meta, **fmt}
        method = "metadata"
    if method:
        nx, ny = int(fmt["nx"]), int(fmt["ny"])
        nbytes, size = nx * ny * np.dtype(fmt["dtype"]).itemsize, path.stat().st_size
        # unknown header length: the data are the last nx*ny values of the file
        off = int(fmt["header_bytes"]) if fmt.get("header_bytes") is not None else size - nbytes
        if off < 0 or off + nbytes > size:
            raise ValueError(f"{path.name}: {nx}x{ny} {fmt['dtype']} at offset {off} does not fit "
                             f"the file ({size} bytes)")
        z = _load(path, fmt["dtype"], off, nx * ny).reshape(ny, nx)
        info.update({"read_method": method, "offset": off, "dtype": fmt["dtype"], "nx": nx, "ny": ny})
    else:
        cands = brute_force(path, rc, verbose)
        if not cands:
            raise ValueError(f"{path.name}: RAW format unresolved (no metadata, not text, no "
                             "plausible offset/dtype/shape); fill the raw: block of config/file_layout.yaml")
        c = cands[0]
        z = _load(path, c["dtype"], c["offset"], c["nx"] * c["ny"]).reshape(c["ny"], c["nx"])
        info.update({"read_method": "brute_force", "offset": c["offset"], "dtype": c["dtype"],
                     "nx": c["nx"], "ny": c["ny"], "norm_tv": c["tv"]})
    if rc.get("flip_rows"):
        z = z[::-1]
    zs, info["z_scale_source"] = ((rc["z_scale_um"], "file_layout") if rc.get("z_scale_um") is not None
                                  else (meta["z_scale_um"], "metadata") if "z_scale_um" in meta
                                  else (None, None))
    if zs is None:
        zs, plausible = _guess_z_scale(z, rc.get("plausible_range_um", [0.1, 100.0]))
        info["z_scale_source"] = "unit_guess" if plausible else "unknown"
    z = z * float(zs)
    info["z_scale_um"] = float(zs)
    ny, nx = z.shape
    px, info["pixel_source"] = ((rc["pixel_um"], "file_layout") if rc.get("pixel_um")
                                else (meta["pixel_um"], "metadata") if meta.get("pixel_um")
                                else (None, "fov_fallback"))
    dx, dy = (float(px), float(px)) if px else (fov[0] / nx, fov[1] / ny)
    hm = HeightMap(z=z, dx_um=dx, dy_um=dy, source=str(path), zmin=float(np.nanmin(z)),
                   zmax=float(np.nanmax(z)), objective=rc.get("objective"),
                   masked_frac=float(np.isnan(z).mean()), info=info)
    # validation: PNG preview of the same scan, else a plausible levelled range
    preview = companion(path, (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    v = validate_against_preview(hm, preview, cfg) if preview else {}
    if np.isfinite(v.get("preview_r", np.nan)):
        info.update(v)
        info["validated"] = bool(v["preview_r"] > rc.get("preview_min_r", 0.9))
    else:
        info.update(v)
        from .wear import level_plane
        zl = level_plane(z, 1)
        span = float(np.nanpercentile(zl, 99.5) - np.nanpercentile(zl, 0.5))
        lo, hi = rc.get("plausible_range_um", [0.1, 100.0])
        info["levelled_range_um"] = span
        info["validated"] = bool(lo <= span <= hi)
    return hm
