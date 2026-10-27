"""Wear from white-light-interferometer height maps: the "band" method with robust levelling.

The plates are rough (Sa 3-9 um, long-wavelength waviness); the worn track is a SMOOTHER, slightly sunken band
(~1-2 um deep) with pile-up ridges at its edges. It is found by roughness, not by depth. The track always runs
along x (rows), so the across-track profile is a per-row statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d


# --------------------------------------------------------------------------------------------- levelling
def fit_plane(z: np.ndarray, mask: np.ndarray | None = None, clip_sigma: float | None = 2.0, passes: int = 3):
    """Least-squares plane z = c0 + c1 x + c2 y on `mask`.

    Robust (clip_sigma set): `passes` clipping passes, each dropping pixels whose residual is more than
    clip_sigma x SD from the median residual, then a final fit (pits, debris and deep texture valleys no
    longer tilt the plane). clip_sigma=None gives one plain least-squares fit.
    Returns (plane, used_mask).
    """
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx - nx / 2) / nx
    y = (yy - ny / 2) / ny
    base = np.isfinite(z) if mask is None else (mask & np.isfinite(z))
    use = base.copy()
    n_clip = passes if clip_sigma is not None else 0
    for p in range(n_clip + 1):
        A = np.column_stack([np.ones(use.sum()), x[use], y[use]])
        coef, *_ = np.linalg.lstsq(A, z[use], rcond=None)
        if p == n_clip:
            break
        res = z - (coef[0] + coef[1] * x + coef[2] * y)
        sd = np.nanstd(res[use])
        use = use & (np.abs(res - np.nanmedian(res[use])) < clip_sigma * sd)
    plane = coef[0] + coef[1] * x + coef[2] * y
    return plane, use


def clipped_row_mean(z: np.ndarray, clip_sigma: float | None = 2.0) -> np.ndarray:
    """Mean of each row; with clip_sigma, values beyond k SD from the row median are dropped (pits, debris)."""
    with np.errstate(all="ignore"):
        if clip_sigma is None:
            return np.nanmean(z, axis=1)
        med = np.nanmedian(z, axis=1, keepdims=True)
        sd = np.nanstd(z, axis=1, keepdims=True)
        return np.nanmean(np.where(np.abs(z - med) <= clip_sigma * sd, z, np.nan), axis=1)


def nan_moving_mean(z: np.ndarray, size: int, axis: int = 1) -> np.ndarray:
    valid = np.isfinite(z)
    num = uniform_filter1d(np.where(valid, z, 0.0), size, axis=axis, mode="nearest")
    den = uniform_filter1d(valid.astype(float), size, axis=axis, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def _fill_nan_1d(v: np.ndarray) -> np.ndarray:
    v = v.copy()
    bad = ~np.isfinite(v)
    if bad.all():
        return np.zeros_like(v)
    if bad.any():
        idx = np.arange(len(v))
        v[bad] = np.interp(idx[bad], idx[~bad], v[~bad])
    return v


# --------------------------------------------------------------------------------------------- result
@dataclass
class BandResult:
    track_found: bool
    valid: bool  # reference unworn on both sides
    flag: str  # ok | no_track | one_sided
    contrast: float = np.nan
    R_unworn: float = np.nan
    R_track: float = np.nan
    core: tuple[int, int] | None = None  # rows [a, b)
    band: tuple[int, int] | None = None
    width_um: float = np.nan
    groove_um2: float = np.nan
    pileup_um2: float = np.nan
    depth_max_um: float = np.nan
    u_um2: float = np.nan
    lod_um2: float = np.nan
    detected: bool = False
    unworn_left_um: float = np.nan
    unworn_right_um: float = np.nan
    sa_unworn_um: float = np.nan
    sd_ref_profile_um: float = np.nan
    groove_ta_rule_um2: float = np.nan  # TA depth rule applied to this profile (C2)
    arrays: dict = field(default_factory=dict, repr=False)

    def scalars(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "arrays"}
        d["core"] = None if self.core is None else list(self.core)
        d["band"] = None if self.band is None else list(self.band)
        return d


# --------------------------------------------------------------------------------------------- method
def analyse_scan(z: np.ndarray, pixel_um: float, cfg: dict, robust: bool = True, keep_arrays: bool = False) -> BandResult:
    """Band method on one height map (rows along the track). `robust=False` = plain levelling (C2 comparison)."""
    clip = cfg["clip_sigma"] if robust else None
    passes = cfg["level_passes"]
    ny, nx = z.shape
    dy = pixel_um

    # 1. first levelling (whole map)
    plane1, _ = fit_plane(z, None, clip, passes)
    z1 = z - plane1

    # 2. local roughness per row: high-pass along the track, SD per row, smoothed across rows
    if cfg.get("highpass_mode", "row") == "box":
        hp = z1 - nan_moving_mean(nan_moving_mean(z1, cfg["highpass_px"], axis=1), cfg["highpass_px"], axis=0)
    else:
        hp = z1 - nan_moving_mean(z1, cfg["highpass_px"], axis=1)
    rough = _fill_nan_1d(np.nanstd(hp, axis=1))
    rough_s = gaussian_filter1d(rough, cfg["smooth_sigma_rows"], mode="nearest")

    # 3. contrast
    n_out = max(1, int(round(cfg["outer_frac"] * ny)))
    R_unworn = float(np.median(np.r_[rough_s[:n_out], rough_s[-n_out:]]))
    R_track = float(np.percentile(rough_s, cfg["track_percentile"]))
    contrast = (R_unworn - R_track) / R_unworn if R_unworn > 0 else 0.0
    arrays = {"rough": rough_s, "z1": z1} if keep_arrays else {}
    if contrast < cfg["min_contrast"]:
        return BandResult(False, False, "no_track", contrast, R_unworn, R_track, arrays=arrays)

    # 4. core (contiguous smooth rows around the smoothest row) and band
    thr = R_track + cfg["core_threshold"] * (R_unworn - R_track)
    i0 = int(np.argmin(rough_s))
    a = i0
    while a > 0 and rough_s[a - 1] < thr:
        a -= 1
    b = i0 + 1
    while b < ny and rough_s[b] < thr:
        b += 1
    core_w = b - a
    width_um = core_w * dy
    ext = int(round(cfg["band_expand"] * core_w))
    ba, bb = max(0, a - ext), min(ny, b + ext)

    # tighten each band edge to the outer foot of its pile-up ridge, i.e. where the smoothed mean profile
    # (levelled on the rows outside the first band) returns to 0; a side without a ridge ends at the core edge
    outside = np.zeros(ny, bool)
    outside[:ba] = True
    outside[bb:] = True
    if outside.sum() >= 5:
        plane_b, _ = fit_plane(z, np.broadcast_to(outside[:, None], z.shape), clip, passes)
        prof1 = clipped_row_mean(z - plane_b, clip)
    else:
        prof1 = clipped_row_mean(z1, clip)
    prof1 = gaussian_filter1d(_fill_nan_1d(prof1), cfg["smooth_sigma_rows"], mode="nearest")
    if a > ba:
        ridge = ba + int(np.argmax(prof1[ba:a]))
        if prof1[ridge] > 0:
            p = ridge
            while p > ba and prof1[p] > 0:
                p -= 1
            ba = p
        else:
            ba = a
    if bb > b:
        ridge = b + int(np.argmax(prof1[b:bb]))
        if prof1[ridge] > 0:
            p = ridge
            while p < bb - 1 and prof1[p] > 0:
                p += 1
            bb = p + 1
        else:
            bb = b

    # 5. reference plane on rows outside band + margin; needs >= min_unworn_um on BOTH sides
    margin = int(np.ceil(cfg["ref_margin_um"] / dy))
    left = np.arange(0, max(0, ba - margin))
    right = np.arange(min(ny, bb + margin), ny)
    unworn_l, unworn_r = len(left) * dy, len(right) * dy
    valid = unworn_l >= cfg["min_unworn_um"] and unworn_r >= cfg["min_unworn_um"]
    res = BandResult(
        True, valid, "ok" if valid else "one_sided", contrast, R_unworn, R_track, (a, b), (ba, bb), width_um,
        unworn_left_um=unworn_l, unworn_right_um=unworn_r,
    )
    ref_rows = np.r_[left, right].astype(int)
    if len(ref_rows) < 5:
        res.arrays = arrays
        return res
    ref_mask = np.zeros_like(z, bool)
    ref_mask[ref_rows, :] = True
    plane2, _ = fit_plane(z, ref_mask, clip, passes)
    z2 = z - plane2

    # 6. profile: per-row mean along the track after 2 sigma clipping of that row
    prof = clipped_row_mean(z2, clip)
    prof = _fill_nan_1d(prof)

    # 7. areas over the band
    pb = prof[ba:bb]
    res.groove_um2 = float(np.sum(np.clip(-pb, 0, None)) * dy)
    res.pileup_um2 = float(np.sum(np.clip(pb, 0, None)) * dy)
    res.depth_max_um = float(np.max(-prof[a:b])) if b > a else np.nan

    # 8. uncertainty and detection limit
    sd_ref = float(np.std(prof[ref_rows]))
    n_eff = max(1.0, (len(ref_rows) * dy) / cfg["corr_length_um"])
    band_w = (bb - ba) * dy
    res.sd_ref_profile_um = sd_ref
    res.u_um2 = sd_ref / np.sqrt(n_eff) * band_w
    res.lod_um2 = cfg["lod_factor"] * res.u_um2
    res.detected = bool(valid and res.groove_um2 > res.lod_um2)
    zr = z2[ref_rows, :]
    res.sa_unworn_um = float(np.nanmean(np.abs(zr - np.nanmean(zr))))

    # TA depth rule (points shallower than max(2.5 SD, 0.25 um) are zeroed one by one), for C2
    cutoff = max(cfg["ta_cutoff_sd"] * sd_ref, cfg["ta_cutoff_min_um"])
    d = np.clip(-pb, 0, None)
    res.groove_ta_rule_um2 = float(np.sum(np.where(d >= cutoff, d, 0.0)) * dy)
    if keep_arrays:
        arrays.update({"z2": z2, "profile": prof, "profile_first": prof1, "ref_rows": ref_rows})
    res.arrays = arrays
    return res
