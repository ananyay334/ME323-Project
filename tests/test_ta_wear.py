"""tribo_extract.ta_wear (the TA's method with its bugs fixed) on synthetic profiles."""
import numpy as np
import pytest

from tribo_extract.ta_wear import groove_metrics, level_plane, orient


def _map(depth=1.0, width_rows=120, ny=600, nx=960, tilt_x=0.0, seed=0, center=300):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.05, (ny, nx))
    y = np.arange(ny)[:, None]
    groove = depth * np.clip(1 - ((y - center) / (width_rows / 2)) ** 2, 0, None)     # parabolic groove
    z = z - groove + tilt_x * np.arange(nx)[None, :] / nx
    return z


def test_full_area_kept_for_a_shallow_groove():
    """The original zeroes points shallower than its cutoff; the fixed version integrates the full area."""
    z, px = _map(depth=0.4), 1.875
    zl, _ = level_plane(orient(z, "x"))
    m = groove_metrics(zl, px)
    true_area = 2 / 3 * 0.4 * 120 * px
    assert m["detected"] and m["area_groove_um2"] == pytest.approx(true_area, rel=0.15)
    assert not m["one_sided"]


def test_orientation_not_decided_by_tilt():
    """A strong tilt along the track made the original transpose the map; track_axis='x' keeps it."""
    z = _map(depth=1.0, tilt_x=30.0)
    assert np.nanstd(np.nanmean(z, axis=0)) > np.nanstd(np.nanmean(z, axis=1))     # original would transpose
    m = groove_metrics(level_plane(orient(z, "x"))[0], 1.875)
    assert m["area_groove_um2"] == pytest.approx(2 / 3 * 1.0 * 120 * 1.875, rel=0.1)
    m_auto = groove_metrics(level_plane(orient(z, "auto"))[0], 1.875)
    assert m_auto["area_groove_um2"] == pytest.approx(m["area_groove_um2"], rel=1e-6)


def test_track_into_shoulders_is_one_sided():
    z = _map(depth=1.0, width_rows=500)
    m = groove_metrics(level_plane(orient(z, "x"))[0], 1.875)
    assert m["one_sided"]


def test_pileup_only_next_to_the_groove():
    z = _map(depth=1.0)
    z[:60] += 0.3                                   # raised texture far from the groove (inside a shoulder band)
    m = groove_metrics(level_plane(orient(z, "x"))[0], 1.875)
    assert m["area_pileup_um2"] < 0.05 * m["area_groove_um2"] + 5
