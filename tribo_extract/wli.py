"""Reconstruct a height map (um) from the WLI software screenshot.

The screenshot shows the height map rendered with a rainbow colour bar
(violet = low ... red = high) that is also *shaded* by the optical intensity,
so pixel RGB values are darker than the colour bar. Hue is unaffected by that
shading, so each pixel's hue is mapped back onto the colour bar to recover its
height. Pixels that are not colour-bar colours (profile lines, markers, text)
are masked out as NaN.

Colour-bar limits are read with OCR when available (mid & max labels; the
min label is often cut off, and min = 2*mid - max). They can always be supplied
from the run sheet instead (cbar_min_um / cbar_max_um), which takes priority.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps
from scipy import ndimage as ndi


@dataclass
class HeightMap:
    z: np.ndarray                 # (ny, nx) heights in um, NaN where masked
    dx_um: float
    dy_um: float
    source: str
    zmin: float = float("nan")
    zmax: float = float("nan")
    objective: str | None = None
    clip_frac_low: float = 0.0
    clip_frac_high: float = 0.0
    masked_frac: float = 0.0
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OCR helpers (optional dependency)
# ---------------------------------------------------------------------------
def _ocr(img: Image.Image, box, whitelist="0123456789.-", psm=7, scale=4) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        c = img.crop(box).convert("L")
        c = c.resize((max(c.width, 1) * scale, max(c.height, 1) * scale), Image.LANCZOS)
        c = ImageOps.invert(c)
        return pytesseract.image_to_string(
            c, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}").strip()
    except Exception:          # tesseract binary missing, etc.
        return ""


def _ocr_number(img: Image.Image, box) -> float | None:
    """OCR a small numeric label printed with two decimals (e.g. 48.89).

    Tries several scales / crops and majority-votes. The decimal point is tiny
    and often dropped by OCR, so a digit string without one gets it re-inserted
    before the last two digits.
    """
    reads = []
    for scale in (3, 4, 5, 6):
        for dy0, dy1 in ((0, 0), (2, -1), (1, 0)):
            b = (box[0], box[1] + dy0, box[2], box[3] + dy1)
            t = _ocr(img, b, scale=scale).replace(" ", "")
            m = re.search(r"-?\d+\.\d{2}", t)
            if m:
                reads.append(float(m.group()))
            elif re.fullmatch(r"-?\d{3,}", t):
                reads.append(float(t[:-2] + "." + t[-2:]))
            if reads and reads.count(reads[-1]) >= 3:        # early stop on 3 agreeing reads
                return reads[-1]
    if not reads:
        return None
    vals, counts = np.unique(reads, return_counts=True)
    return float(vals[np.argmax(counts)])


def _to_float(s: str):
    m = re.search(r"-?\d+\.\d+|-?\d+", s or "")
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------
def _components(mask: np.ndarray):
    lab, n = ndi.label(ndi.binary_closing(mask, iterations=2))
    objs = ndi.find_objects(lab)
    out = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        fill = (lab[sl] == i).mean()
        out.append({"y0": sl[0].start, "y1": sl[0].stop, "x0": sl[1].start, "x1": sl[1].stop,
                    "h": h, "w": w, "fill": fill, "area": h * w})
    return out


def _find_layout(hsv: np.ndarray):
    sat = (hsv[..., 1] > 0.35) & (hsv[..., 2] > 0.15)
    comps = _components(sat)
    maps = [c for c in comps if c["fill"] > 0.5 and c["h"] > 100 and c["w"] > 100]
    if not maps:
        raise ValueError("could not locate the height-map panel in the screenshot")
    m = max(maps, key=lambda c: c["area"])
    # trim border rows/cols that the morphological closing glued on (scale labels)
    sub = sat[m["y0"]:m["y1"], m["x0"]:m["x1"]]
    rows = np.where(sub.mean(axis=1) > 0.5)[0]
    cols = np.where(sub.mean(axis=0) > 0.5)[0]
    m = dict(m, y0=m["y0"] + rows[0], y1=m["y0"] + rows[-1] + 1,
             x0=m["x0"] + cols[0], x1=m["x0"] + cols[-1] + 1)
    bars = [c for c in comps if c["y0"] >= m["y1"] and c["w"] > 15 * max(c["h"], 1)
            and c["w"] > 0.5 * m["w"]]
    if not bars:
        raise ValueError("could not locate the colour bar below the height map")
    b = max(bars, key=lambda c: c["w"])
    return m, b


def _colorbar_lut(hsv: np.ndarray, b: dict):
    band = hsv[b["y0"]:b["y1"], b["x0"]:b["x1"]]
    good_rows = [(band[r, :, 1] > 0.5).mean() for r in range(band.shape[0])]
    rows = [r for r, g in enumerate(good_rows) if g > 0.8 * max(good_rows)]
    row = band[rows[len(rows) // 2]]
    cols = np.where(row[:, 1] > 0.5)[0]
    # keep the longest contiguous run (drops the grey slider handles)
    splits = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
    cols = max(splits, key=len)
    hue = np.array([np.median(band[rows, c, 0]) for c in cols]) * 360.0
    x0, x1 = b["x0"] + cols[0], b["x0"] + cols[-1]
    return hue, x0, x1


def _hue_cut(hue_bar: np.ndarray) -> float:
    """Circular hue value that lies in the largest gap of the colour bar."""
    h = np.sort(np.mod(hue_bar, 360))
    gaps = np.diff(np.concatenate([h, [h[0] + 360]]))
    i = int(np.argmax(gaps))
    return float(np.mod(h[i] + gaps[i] / 2, 360))


def _unwrap(h: np.ndarray, cut: float) -> np.ndarray:
    return np.where(h > cut, h - 360.0, h)


def read_wli_screenshot(path: str | Path, cfg: dict, cbar_min=None, cbar_max=None,
                        fov_um=None) -> HeightMap:
    w = cfg["wli"]
    path = Path(path)
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img).astype(float) / 255.0
    hsv = rgb_to_hsv(rgb)
    m, b = _find_layout(hsv)
    hue_bar, bx0, bx1 = _colorbar_lut(hsv, b)
    info = {"map_bbox": [m["x0"], m["y0"], m["x1"], m["y1"]], "cbar_x": [int(bx0), int(bx1)],
            "cbar_y": [b["y0"], b["y1"]]}

    # ----- colour-bar limits -----------------------------------------------
    ocr = {}
    if w.get("use_ocr", True):
        ly0, ly1 = b["y1"], min(b["y1"] + 16, img.height)
        cx = (bx0 + bx1) // 2
        ocr["left"] = _ocr_number(img, (max(b["x0"] - 20, 0), ly0, b["x0"] + 60, ly1))
        ocr["mid"] = _ocr_number(img, (cx - 42, ly0, cx + 42, ly1))
        ocr["right"] = _ocr_number(img, (b["x1"] - 60, ly0, min(b["x1"] + 25, img.width), ly1))
        foot = _ocr(img, (b["x1"] + 5, b["y1"] - 8, img.width, img.height), psm=7,
                    whitelist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. ")
        ocr["footer"] = foot
    zmin_ocr = zmax_ocr = None
    if ocr.get("mid") is not None and ocr.get("right") is not None:
        zmax_ocr = ocr["right"]
        zmin_ocr = 2 * ocr["mid"] - ocr["right"]
        if ocr.get("left") is not None and abs(ocr["left"] - zmin_ocr) < 0.02:
            zmin_ocr = ocr["left"]
    if zmin_ocr is not None and not (zmin_ocr < ocr["mid"] < zmax_ocr):
        info["warn_cbar"] = "OCR colour-bar labels inconsistent; ignored"
        zmin_ocr = zmax_ocr = None
    info["ocr"] = ocr
    zmin = float(cbar_min) if cbar_min is not None and np.isfinite(cbar_min) else zmin_ocr
    zmax = float(cbar_max) if cbar_max is not None and np.isfinite(cbar_max) else zmax_ocr
    if zmin is None or zmax is None:
        raise ValueError(
            f"{path.name}: colour-bar limits unknown. Install tesseract+pytesseract or fill "
            "cbar_min_um / cbar_max_um for this run in data/run_sheet.csv")
    if zmin_ocr is not None and cbar_min is not None and np.isfinite(cbar_min) \
            and abs(zmin_ocr - cbar_min) > 0.02:
        info["warn_cbar"] = f"run sheet cbar_min {cbar_min} differs from OCR {zmin_ocr:.2f}"

    # ----- objective / field of view ---------------------------------------
    ms = re.search(r"Run-\d{4}-\d{2}-\d{2}-\d{6}", ocr.get("footer", "") or "")
    info["scan_id"] = ms.group() if ms else None
    objective = None
    mo = re.search(r"(\d{1,3})\s*[xX]", ocr.get("footer", "") or "")
    if mo:
        objective = f"{mo.group(1)}x"
    if fov_um is None:
        key = objective if objective in w["fov_um"] else w["default_objective"]
        fov_um = w["fov_um"][key]
        objective = objective or key

    # ----- map pixels -> height --------------------------------------------
    sub = hsv[m["y0"]:m["y1"], m["x0"]:m["x1"]]
    cut = _hue_cut(hue_bar)
    hb = _unwrap(hue_bar, cut)
    pos = np.linspace(0.0, 1.0, len(hb))
    if hb[0] > hb[-1]:                           # make hue increasing for interp
        hb, pos = hb[::-1], pos[::-1]
    hb = np.maximum.accumulate(hb)               # enforce monotonic
    hp = _unwrap(sub[..., 0] * 360.0, cut)
    frac = np.interp(hp, hb, pos)

    val = sub[..., 2]
    mask = (sub[..., 1] < w["min_saturation"]) | (val < w["min_value"])
    mask |= (hp < hb.min() - w["hue_margin_deg"]) | (hp > hb.max() + w["hue_margin_deg"])
    mask |= (val - ndi.median_filter(val, size=7)) > w["bright_overlay_dv"]
    if w.get("mask_dilate_px", 0):
        mask = ndi.binary_dilation(mask, iterations=int(w["mask_dilate_px"]))

    z = zmin + frac * (zmax - zmin)
    z[mask] = np.nan
    valid = ~mask
    ny, nx = z.shape
    hm = HeightMap(
        z=z, dx_um=fov_um[0] / nx, dy_um=fov_um[1] / ny, source=str(path),
        zmin=zmin, zmax=zmax, objective=objective,
        clip_frac_low=float(((frac <= 0.003) & valid).sum() / valid.sum()),
        clip_frac_high=float(((frac >= 0.997) & valid).sum() / valid.sum()),
        masked_frac=float(mask.mean()), info=info)
    hm.info["fov_um"] = list(fov_um)
    return hm


def read_height_matrix(path: str | Path, fov_um) -> HeightMap:
    """Load a raw height export (plain numeric grid in um: .csv/.txt/.asc).

    Preferred over the screenshot whenever the TA can export it - no colour
    quantisation, no overlays and no clipping.
    """
    path = Path(path)
    txt = path.read_text(errors="replace").splitlines()
    rows = []
    for ln in txt:
        parts = re.split(r"[,\t; ]+", ln.strip())
        try:
            rows.append([float(p) for p in parts if p != ""])
        except ValueError:
            continue                              # header / comment line
    n = max(len(r) for r in rows)
    z = np.array([r + [np.nan] * (n - len(r)) for r in rows if len(r) == n], float)
    ny, nx = z.shape
    return HeightMap(z=z, dx_um=fov_um[0] / nx, dy_um=fov_um[1] / ny, source=str(path),
                     zmin=float(np.nanmin(z)), zmax=float(np.nanmax(z)))
