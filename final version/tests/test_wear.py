import numpy as np

from tribo_final.extract.wear_band import analyse_scan, clipped_row_mean, fit_plane

from conftest import synthetic_surface

PX = 1.875


def test_band_method_on_six_synthetic_surfaces(cfg):
    gb, wb = [], []
    for seed in range(6):
        z, prof = synthetic_surface(seed)
        r = analyse_scan(z, PX, cfg["wear"])
        assert r.track_found and r.valid, (seed, r.flag)
        assert r.detected
        true_groove = np.sum(np.clip(-prof, 0, None)) * PX
        true_width = 200 * PX
        gb.append(abs(r.groove_um2 / true_groove - 1))
        wb.append(abs(r.width_um / true_width - 1))
    assert np.median(gb) < 0.20, gb
    assert np.median(wb) < 0.10, wb


def test_no_track_on_unworn_surface(cfg):
    for seed in range(3):
        z, _ = synthetic_surface(100 + seed, band=None)
        r = analyse_scan(z, PX, cfg["wear"])
        assert not r.track_found and r.flag == "no_track"


def test_one_sided_reference_flagged(cfg):
    z, _ = synthetic_surface(7, band=(30, 330))
    r = analyse_scan(z, PX, cfg["wear"])
    assert r.track_found
    assert not r.valid and r.flag == "one_sided"
    assert not r.detected


def test_robust_levelling_ignores_a_25um_pit():
    rng = np.random.default_rng(3)
    ny, nx = 600, 960
    yy, xx = np.mgrid[0:ny, 0:nx]
    true = 0.004 * xx - 0.006 * yy + 2.0
    z = true + rng.normal(0, 0.5, (ny, nx))
    z[(yy - 120) ** 2 + (xx - 300) ** 2 < 40 ** 2] -= 25.0
    robust, _ = fit_plane(z, None, clip_sigma=2.0, passes=3)
    plain, _ = fit_plane(z, None, clip_sigma=None)
    assert np.max(np.abs(robust - true)) < 0.1
    assert np.max(np.abs(plain - true)) > 3 * np.max(np.abs(robust - true))
    # the clipped row mean through the pit rows stays on the surface
    rows = clipped_row_mean(z - robust, 2.0)
    assert np.max(np.abs(rows[90:150])) < 0.2
