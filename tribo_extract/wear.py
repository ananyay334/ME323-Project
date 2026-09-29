"""Wear-scar measurement on a WLI height map.

Method (ASTM G133-style for linear reciprocating tests):
  1. level the map with a robust plane fit;
  2. find the sliding direction (or take it from the run sheet);
  3. rotate so the track is horizontal and average along the track
     -> one low-noise cross-section profile;
  4. fit a reference line to the unworn shoulders and integrate the
     depression below it -> worn cross-section area A (um^2);
  5. wear volume V = A * stroke length.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi

from .wli import HeightMap


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _poly_terms(xx, yy, order):
    cols = [np.ones_like(xx)]
    for d in range(1, order + 1):
        for i in range(d + 1):
            cols.append(xx ** (d - i) * yy ** i)
    return cols


def fit_form(z: np.ndarray, use: np.ndarray, order: int = 2) -> np.ndarray:
    """Least-squares polynomial surface (form) fitted to z[use]; returns the surface."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xx = (xx - nx / 2) / nx
    yy = (yy - ny / 2) / nx
    terms = _poly_terms(xx, yy, order)
    A = np.stack([t[use] for t in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, z[use], rcond=None)
    return sum(c * t for c, t in zip(coef, terms))


def level_plane(z: np.ndarray, order: int = 1, iters: int = 3) -> np.ndarray:
    """Remove a robust polynomial form (order 1 = plane, 2 = quadratic)."""
    ok = np.isfinite(z)
    for _ in range(iters):
        res = z - fit_form(z, ok, order)
        s = np.nanstd(res[ok])
        ok = np.isfinite(z) & (np.abs(res) < 2.5 * s)
    return z - fit_form(z, ok, order)


def _nan_zoom(z: np.ndarray, zoom) -> np.ndarray:
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nanmean(z))
    zf = ndi.zoom(filled, zoom, order=1)
    vf = ndi.zoom(valid.astype(float), zoom, order=1)
    zf[vf < 0.5] = np.nan
    return zf


def rotate_nan(z: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate so that a track at `angle_deg` (CCW from +x, as displayed) becomes horizontal."""
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nanmean(z))
    zr = ndi.rotate(filled, -angle_deg, reshape=True, order=1, mode="constant", cval=np.nan)
    vr = ndi.rotate(valid.astype(float), -angle_deg, reshape=True, order=1, mode="constant", cval=0)
    zr[(vr < 0.5) | ~np.isfinite(zr)] = np.nan
    return zr


def _profile(zr: np.ndarray, min_cov: float):
    cov = np.isfinite(zr).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        P = np.nanmean(zr, axis=1)
    keep = cov >= min_cov * cov.max()
    return P, cov, keep


def find_track_angle(z: np.ndarray, step: float = 2.0, min_cov: float = 0.5) -> float:
    zs = _nan_zoom(z, 0.5)                          # speed: half resolution
    def score(a):
        P, _, keep = _profile(rotate_nan(zs, a), min_cov)
        p = P[keep]
        p = p[np.isfinite(p)]
        if len(p) < 20:
            return -np.inf
        x = np.arange(len(p))
        p = p - np.polyval(np.polyfit(x, p, 1), x)
        return float(np.std(p))
    coarse = np.arange(0, 180, step)
    s = [score(a) for a in coarse]
    best = coarse[int(np.argmax(s))]
    fine = np.arange(best - step, best + step + 1e-9, step / 8)
    s = [score(a) for a in fine]
    return float(np.mod(fine[int(np.argmax(s))], 180))


def remove_along_track_form(zr: np.ndarray, order: int = 2) -> np.ndarray:
    """Remove tilt/curvature along the track (x in the rotated frame) only.

    Model: z(row, u) = f(row) + b*u + c*u^2, with a free offset f per row (that
    offset *is* the cross-track profile incl. the groove, so it is untouched).
    Without this, a small along-track tilt biases rows that are only partially
    covered (the corners of the rotated map) and distorts the profile ends.
    """
    ny, nx = zr.shape
    u = (np.arange(nx) - nx / 2) / nx
    U = [np.broadcast_to(u ** k, zr.shape) for k in range(1, order + 1)]
    ok = np.isfinite(zr)
    cnt = ok.sum(axis=1, keepdims=True).clip(min=1)
    def rowdemean(a):
        a = np.where(ok, a, 0.0)
        return np.where(ok, a - a.sum(axis=1, keepdims=True) / cnt, 0.0)
    zd = rowdemean(zr)
    Ud = [rowdemean(t) for t in U]
    A = np.stack([t[ok] for t in Ud], axis=1)
    coef, *_ = np.linalg.lstsq(A, zd[ok], rcond=None)
    return zr - sum(c * t for c, t in zip(coef, U))


def _upper_sigma(r):
    """Noise estimate from the positive residuals only (a groove only adds negatives)."""
    pos = r[r > 0]
    return float(np.sqrt(np.mean(pos ** 2))) if len(pos) > 5 else float(np.std(r))


def _als_line(x, p, iters=30, init_top_frac=None):
    """Straight 'surface' line that ignores depressions.

    Asymmetric iteratively-reweighted fit (the idea behind asymmetric-least-squares
    baseline correction): points far *below* the line (a groove) get ~zero weight,
    points above keep full weight, so the line follows the unworn surface.
    init_top_frac starts from the highest points only - needed when a wide, deep
    scar fills most of the profile and an ordinary first fit runs through its middle.
    """
    if init_top_frac:
        d = p - np.polyval(np.polyfit(x, p, 1), x)
        wts = (d >= np.quantile(d, 1 - init_top_frac)).astype(float)
    else:
        wts = np.ones_like(p)
    for _ in range(iters):
        coef = np.polyfit(x, p, 1, w=np.sqrt(np.maximum(wts, 1e-3)))
        r = p - np.polyval(coef, x)
        keep = wts > 0.5
        if init_top_frac:
            s = float(np.sqrt(np.mean(r[keep] ** 2))) * 1.5
        else:
            s = _upper_sigma(r[keep]) if keep.sum() > 10 else _upper_sigma(r)
        new = np.where(r < -2.0 * s, 1e-3, 1.0)
        if np.array_equal(new, wts):
            break
        wts = new
    return np.polyval(coef, x)


def _choose_reference(x, p, m, max_order, min_shoulder_frac, corr_len_px):
    """Fit the reference on shoulder points m. Curvature (order 2) is used only if
    both shoulders are long enough AND an F-test (on an effective sample size that
    accounts for the correlated surface texture) says the curvature is real -
    otherwise texture would be mistaken for form and bias the area."""
    n = len(p)
    idx = np.where(m)[0]
    fit1 = np.polyval(np.polyfit(x[m], p[m], 1), x)
    if max_order < 2 or len(idx) < 10:
        return fit1, 1
    # split shoulders around the excluded block (or around the middle if nothing excluded)
    gap = np.where(~m)[0]
    mid = gap.mean() if len(gap) else n / 2
    left, right = (idx < mid).sum(), (idx > mid).sum()
    if min(left, right) < min_shoulder_frac * n:
        return fit1, 1
    fit2 = np.polyval(np.polyfit(x[m], p[m], 2), x)
    ms1 = np.mean((p[m] - fit1[m]) ** 2)
    ms2 = np.mean((p[m] - fit2[m]) ** 2)
    n_eff = max(len(idx) / max(corr_len_px, 1.0), 4.0)
    F = (ms1 - ms2) / max(ms2, 1e-18) * (n_eff - 3)
    return (fit2, 2) if F > 10.0 else (fit1, 1)


def _core(r, sig, k_sigma, core_frac=0.5):
    """Deepest contiguous region: points deeper than core_frac * max depth."""
    depth = -r.min()
    if depth < k_sigma * sig:
        return None
    lab, _ = ndi.label(r < -core_frac * depth)
    i = lab[int(np.argmin(r))]
    idx = np.where(lab == i)[0]
    return int(idx[0]), int(idx[-1]), float(depth)


def _edges(r, cL, cR, thr):
    n = len(r)
    L, R = cL, cR
    while L > 0 and r[L - 1] < -thr:
        L -= 1
    while R < n - 1 and r[R + 1] < -thr:
        R += 1
    return L, R


def detect_groove(v: np.ndarray, P: np.ndarray, k_sigma: float, smooth_px: float,
                  edge_k: float = 1.0, ref_order: int = 2, min_shoulder_frac: float = 0.15,
                  corr_len_um: float = 20.0, edge_depth_frac: float = 0.10):
    """Locate the wear groove in an along-track-averaged cross-section profile.

    1. a straight reference that ignores depressions (asymmetric fit);
    2. core = the deepest region below half the maximum depth (must be deeper than
       k_sigma * noise) - insensitive to shallow specimen form around the scar;
    3. the reference is refitted on everything outside core +/- 40 % of its width
       (with curvature only when both shoulders are long and the F-test supports it);
    4. edges = where the depth returns above max(edge_k * noise, 10 % of max depth);
       for a parabolic scar the tails beyond such a cut hold < 1 % of the area;
    5. repeat 2-4 until stable; area = depression below the reference between edges.
    """
    Ps = ndi.gaussian_filter1d(P, smooth_px) if smooth_px > 0 else P.copy()
    dv = float(np.median(np.diff(v)))
    n = len(Ps)
    x = (v - v.mean()) / (np.ptp(v) + 1e-12)
    corr_px = corr_len_um / dv
    ref, order = _als_line(x, Ps), 1
    sig = _upper_sigma(Ps - ref)
    core = _core(Ps - ref, sig, k_sigma)
    if core is None:
        # a scar wider than ~half the profile drags an ordinary first fit into it;
        # if the profile relief is far above the texture level, restart from the top
        hp = Ps - ndi.gaussian_filter1d(Ps, corr_px / 2)
        if np.ptp(ndi.gaussian_filter1d(Ps, corr_px)) > 10 * np.std(hp):
            ref = _als_line(x, Ps, init_top_frac=0.3)
            sig = _upper_sigma(Ps - ref)
            core = _core(Ps - ref, sig, k_sigma)
    if core is None:
        # nothing below a straight surface: allow curvature (specimen form) and retry
        ref, order = _choose_reference(x, Ps, np.ones(n, bool), ref_order, min_shoulder_frac, corr_px)
        sig = _upper_sigma(Ps - ref)
        core = _core(Ps - ref, sig, k_sigma)
    best, prev = None, None
    for _ in range(8):
        if core is None:
            break
        cL, cR, _d = core
        ext = max(int(0.4 * (cR - cL + 1)), 3)
        m = np.ones(n, bool)
        m[max(cL - ext, 0):min(cR + ext + 1, n)] = False
        if m.sum() >= 10:
            ref, order = _choose_reference(x, Ps, m, ref_order, min_shoulder_frac, corr_px)
            r = Ps - ref
            sig = float(np.sqrt(np.mean(r[m] ** 2)))
        else:
            r = Ps - ref
        new_core = _core(r, sig, k_sigma)
        if new_core is None:
            break
        cL, cR, depth = new_core
        L, R = _edges(r, cL, cR, max(edge_k * sig, edge_depth_frac * depth))
        best = {"L": L, "R": R, "ref": ref, "r": r, "sigma": sig, "dv": dv, "Ps": Ps,
                "ref_order": order, "area_um2": float(-r[L:R + 1].clip(max=0).sum() * dv)}
        if (cL, cR) == (core[0], core[1]) or (cL, cR) == prev:
            break
        prev = (core[0], core[1])
        core = new_core
    if best is None:
        r = Ps - ref
        return {"L": None, "R": None, "ref": ref, "r": r, "sigma": _upper_sigma(r),
                "area_um2": 0.0, "dv": dv, "Ps": Ps, "ref_order": order}
    return best


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def measure_wear_scar(hm: HeightMap, cfg: dict, angle_deg=None, expected_width_um=np.nan) -> tuple[dict, dict]:
    w = cfg["wear"]
    z = hm.z
    if abs(hm.dy_um / hm.dx_um - 1) > 1e-3:                 # isotropic pixels
        z = _nan_zoom(z, (hm.dy_um / hm.dx_um, 1.0))
    px = hm.dx_um
    zl = level_plane(z, order=1)

    auto = angle_deg is None or isinstance(angle_deg, str) or not np.isfinite(angle_deg)
    ang = find_track_angle(zl, w["angle_search_step_deg"], w["min_coverage_frac"]) if auto else float(angle_deg)
    order = int(w.get("form_order", 2))
    zr = remove_along_track_form(rotate_nan(zl, ang), order)
    P, cov, keep = _profile(zr, w["min_coverage_frac"])
    rows = np.where(keep & np.isfinite(P))[0]
    v = rows * px
    Pk = P[rows]
    g = detect_groove(v, Pk, w["detect_k_sigma"], w["profile_smooth_px"], w.get("edge_k_sigma", 1.0),
                      ref_order=order)

    flags = []
    # along-track texture amplitude: the row mean (groove + cross-track form) is
    # removed, what is left is roughness. A single scan cannot tell a scar that is
    # shallower than the roughness apart from surface form, so that sets a floor.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        texture = float(np.nanmean(np.abs(zr[rows] - np.nanmean(zr[rows], axis=1, keepdims=True))))
    min_depth = max(w["detect_k_sigma"] * g["sigma"], w.get("min_depth_texture_mult", 1.0) * texture)
    out = {"track_angle_deg": round(ang, 2), "track_angle_source": "auto" if auto else "run_sheet",
           "profile_noise_um": g["sigma"], "texture_Sa_um": texture, "min_detectable_depth_um": min_depth,
           "px_um": px}
    width_ref = np.nanmax([w["min_width_um"], expected_width_um if np.isfinite(expected_width_um) else 0])
    out["area_LOD_um2"] = float(min_depth * width_ref * 2 / 3)          # parabolic scar at the LOD

    detected = False
    if g["L"] is not None:
        L, R = g["L"], g["R"]
        width = (R - L + 1) * g["dv"]
        depth = float(-g["r"][L:R + 1].min())
        detected = width >= w["min_width_um"] and depth >= min_depth
        out.update({"track_width_um": float(width), "track_max_depth_um": depth,
                    "reference_order": g.get("ref_order")})
        if detected and (L == 0 or R == len(v) - 1):
            flags.append("track_touches_fov_edge")
    if not detected:
        hp = g["Ps"] - ndi.gaussian_filter1d(g["Ps"], 10 / g["dv"])
        if np.ptp(ndi.gaussian_filter1d(g["Ps"], 20 / g["dv"])) > 10 * np.std(hp) and \
                np.ptp(g["Ps"]) > 3 * texture:
            flags.append("large_relief_not_resolved_check_scan")
    if detected:
        L, R = g["L"], g["R"]
        r = g["r"]
        hw = max((R - L) // 2, 3)
        pile = r[max(L - hw, 0):L].clip(min=0).sum() + r[R + 1:R + 1 + hw].clip(min=0).sum()
        out["worn_area_um2"] = g["area_um2"]
        out["pileup_area_um2"] = float(pile * g["dv"])
        out["net_area_um2"] = out["worn_area_um2"] - out["pileup_area_um2"]
        # along-track consistency: a real track has a similar cross-section everywhere
        cols = np.where(np.isfinite(zr[rows]).mean(axis=0) > 0.5)[0]
        refline = g["ref"]
        areas = []
        for sgi in np.array_split(cols, w["n_segments"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ps = np.nanmean(zr[rows][:, sgi], axis=1)
            ps = ndi.gaussian_filter1d(np.nan_to_num(ps, nan=np.nanmean(ps)), w["profile_smooth_px"])
            areas.append(-(ps - refline)[L:R + 1].clip(max=0).sum() * g["dv"])
        areas = np.array(areas)
        out["segment_area_cv"] = float(areas.std() / areas.mean()) if areas.mean() > 0 else np.nan
        if out["segment_area_cv"] > w["max_segment_cv"]:
            flags.append("track_nonuniform_along_length")
        if hm.clip_frac_low > cfg["wli"]["clip_frac_warn"]:
            flags.append("colourbar_clipping_low")
    else:
        out["worn_area_um2"] = 0.0
        flags.append("no_wear_track_detected")
    out["wear_detected"] = bool(detected)
    out["wear_flags"] = ";".join(flags)

    internals = {"zl": zl, "zr": zr, "v": v, "P": Pk, "g": g, "angle": ang}
    return out, internals


# ---------------------------------------------------------------------------
# baseline (t = 0 s) subtraction
# ---------------------------------------------------------------------------
def register_shift(a: np.ndarray, b: np.ndarray, max_shift_px: int = 60) -> tuple[int, int]:
    """Integer (dy, dx) that best aligns b onto a (phase correlation)."""
    fa = np.where(np.isfinite(a), a - np.nanmean(a), 0.0)
    fb = np.where(np.isfinite(b), b - np.nanmean(b), 0.0)
    ny, nx = min(fa.shape[0], fb.shape[0]), min(fa.shape[1], fb.shape[1])
    fa, fb = fa[:ny, :nx], fb[:ny, :nx]
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.fft.fft2(fa * win) * np.conj(np.fft.fft2(fb * win))
    corr = np.real(np.fft.ifft2(F / (np.abs(F) + 1e-12)))
    corr = np.fft.fftshift(corr)
    cy, cx = ny // 2, nx // 2
    sub = corr[cy - max_shift_px:cy + max_shift_px + 1, cx - max_shift_px:cx + max_shift_px + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    return int(iy - max_shift_px), int(ix - max_shift_px)


def subtract_baseline(worn: HeightMap, base: HeightMap, max_shift_px: int = 60):
    """Height change map (worn - baseline) after aligning the two scans.

    With the in-situ WLI the specimen is not unmounted, so both scans image the
    same spot; the subtraction removes surface texture and form, leaving only the
    material removed (and any pile-up). Returns (HeightMap, info dict).
    """
    a, b = level_plane(worn.z, 1), level_plane(base.z, 1)
    dy, dx = register_shift(a, b, max_shift_px)
    ny, nx = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    bs = np.full((ny, nx), np.nan)
    ys, xs = slice(max(dy, 0), ny + min(dy, 0)), slice(max(dx, 0), nx + min(dx, 0))
    yb, xb = slice(max(-dy, 0), ny + min(-dy, 0)), slice(max(-dx, 0), nx + min(-dx, 0))
    bs[ys, xs] = b[yb, xb]
    diff = a[:ny, :nx] - bs
    hm = HeightMap(z=diff, dx_um=worn.dx_um, dy_um=worn.dy_um, source=f"{worn.source} - baseline",
                   zmin=worn.zmin, zmax=worn.zmax, objective=worn.objective,
                   clip_frac_low=worn.clip_frac_low, clip_frac_high=worn.clip_frac_high,
                   masked_frac=float(np.isnan(diff).mean()), info=dict(worn.info))
    # correlation quality: residual roughness vs original roughness
    info = {"baseline_shift_px": [dy, dx],
            "baseline_residual_ratio": float(np.nanstd(diff) / max(np.nanstd(a), 1e-12))}
    return hm, info
