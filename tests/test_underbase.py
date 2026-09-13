"""Underbase: derived from the colors, choked, gain-compensated on its own, printed first."""

import numpy as np
import pytest

import halftoner as ht
from halftoner.color import hex_to_linear, srgb_to_linear
from halftoner.policy import apply_policy
from halftoner.render.bitmap import hit_rows
from halftoner.render.raster import composite

GARMENT = "#161618"


def _shirt(source, choke=0.3, gain=0.0, weights=None, ruling=40, opacity=0.0, underbase=True, ceiling=150):
    base = ht.Underbase(ink=ht.Ink("white", "#F4F4F0", angle=22.5, opacity=0.9), choke_mm=choke, gain=gain,
                        weights=weights or {}) if underbase else None
    return ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=254),  # 10 px/mm
        inks=ht.InkSet(ht.Ink("gold", "#F2C230", angle=52.5, opacity=opacity, source=source)),
        screen=ht.Screen(ruling_lpi=ruling),
        substrate=ht.Substrate("tee", paper=GARMENT, ruling_ceiling_lpi=ceiling),
        underbase=base,
    )


def _first_inked_column(plate, row_mm=10.0, dpi=254):
    width = round(20 * dpi / 25.4)
    (_, _, _, inked), = hit_rows(plate, 0.0, row_mm, width, 1, dpi)
    cols = np.flatnonzero(inked[0])
    return int(cols[0]) if cols.size else None


def test_opaque_ink_covers_with_its_own_color():
    primaries = ht.InkSet(ht.Ink("white", "#F4F4F0", opacity=1.0)).primaries(GARMENT)
    assert primaries[1] == pytest.approx(hex_to_linear("#F4F4F0"), abs=1e-9)


def test_underbase_prints_first_under_the_colors():
    job = _shirt(ht.Masked(ht.Constant(1.0), ht.Rect(5, 5, 10, 10)))
    assert [p.ink.name for p in job.plates()] == ["white", "gold"]
    assert [i.name for i in job.print_inks] == ["white", "gold"]
    white = job.plates()[0]
    X, Y = white.centers()
    assert np.all(white.area[(X > 7) & (X < 13) & (Y > 7) & (Y < 13)] == 1.0)
    assert np.all(white.area[(X < 3) | (X > 17) | (Y < 3) | (Y > 17)] == 0.0)


def test_region_edges_are_choked_at_full_resolution():
    white, gold = _shirt(ht.Masked(ht.Constant(1.0), ht.Rect(5, 5, 10, 10)), choke=0.3).plates()
    assert _first_inked_column(gold) == 50  # the color starts at x = 5.0 mm
    assert _first_inked_column(white) == 53  # the base starts 0.3 mm inside it, not stair-stepped by cells


def test_tonal_edges_are_eroded_on_the_cell_grid():
    step = ht.Function(lambda x, y: np.where(x < 10, 1.0, 0.0))
    white = _shirt(step, choke=1.0).plates()[0]
    X, Y = white.centers()
    mid, pitch = (Y > 3) & (Y < 17), white.pitch_mm
    assert np.all(white.area[mid & (X > 10 - 1.0 + pitch) & (X < 10)] == 0.0)  # within the choke of the edge
    assert np.all(white.area[mid & (X > 1) & (X < 10 - 1.0 - pitch)] == 1.0)  # clear of it


def test_weights_and_its_own_gain_compensation():
    white = _shirt(ht.Constant(0.5), gain=0.2, weights={"gold": 0.6}).plates()[0]
    X, Y = white.centers()
    on = (X > 2) & (X < 18) & (Y > 2) & (Y < 18)
    assert white.area[on] == pytest.approx(float(ht.Curve.gain(0.2).inverse()(0.3)), abs=1e-3)
    assert white.printed[on] == pytest.approx(0.3, abs=2e-3)  # after its own gain it lands where demanded


def test_underbase_lifts_a_light_color_off_a_dark_garment():
    src = ht.Masked(ht.Constant(1.0), ht.Rect(2, 2, 16, 16))
    luminance = lambda px: srgb_to_linear(px[60:140, 60:140] / 255.0).mean()  # noqa: E731
    bare = luminance(composite(_shirt(src, underbase=False), supersample=2))
    based = luminance(composite(_shirt(src), supersample=2))
    assert based > 5 * bare


def test_underbase_round_trips_reports_and_separates(tmp_path):
    job = _shirt(ht.Masked(ht.Constant(0.6), ht.Rect(2, 2, 16, 16)), gain=0.25, weights={"gold": 0.8})
    job.save(tmp_path / "tee.json")
    back = ht.Recipe.load(tmp_path / "tee.json")
    assert back.to_dict(tmp_path) == job.to_dict(tmp_path)
    assert np.array_equal(back.plates()[0].area, job.plates()[0].area)
    report = str(job.report())
    assert "white + gold" in report and "white/gold angle separation" in report
    counts = job.render_pdf(tmp_path / "tee.pdf", mode="bitmap", bitmap_dpi=300)
    assert set(counts) == {"white", "gold"} and counts["white"] > 0


def test_pod_caps_the_underbase_ruling_too():
    job = _shirt(ht.Constant(0.5), ruling=80, ceiling=55)
    capped, outcome = apply_policy(job, "pod")
    assert set(outcome.capped) == {"white", "gold"}
    assert capped.ruling_for(capped.underbase.ink) == 55


def test_garment_profile_builds_an_underbase():
    p = ht.PressProfile.load("plastisol_dark_garment_nominal")
    job = p.recipe(ht.Canvas.of(20, 20, dpi=100), [("gold", "#F2C230"), ("red", "#C8302A")],
                   source=ht.Constant(0.5), underbase=True)
    assert job.underbase.choke_mm == 0.3 and job.underbase.ink.angle == 22.5
    assert [i.name for i in job.print_inks] == ["underbase", "gold", "red"]
    assert job.inks.inks[0].opacity == 0.8
    assert all(c.ok for c in job.report().checks if c.kind == "angle")


def _choke_check(job):
    (check,) = [c for c in job.report().checks if c.kind == "choke"]
    return check


def test_a_choke_finer_than_a_cell_is_reported_as_doing_nothing():
    # 40 lpi is a 0.635 mm cell, so a 0.3 mm choke cannot erode a tonal edge at all.
    job = _shirt(ht.Constant(0.5), choke=0.3, ruling=40)
    check = _choke_check(job)
    assert not check.ok
    assert check.name == "white choke >= one cell"
    assert "0.3 mm is 0.47 of a 0.635 mm cell" in check.detail
    assert "[report] white choke >= one cell" in str(apply_policy(job, "screen")[1])


def test_a_choke_over_a_cell_passes():
    job = _shirt(ht.Constant(0.5), choke=1.0, ruling=40)  # 1.57 cells
    assert _choke_check(job).ok


def test_the_reported_threshold_is_where_erosion_actually_starts():
    """The check must not pass while erode() is still a no-op, in either direction."""
    step = ht.Function(lambda x, y: np.where(x < 10, 1.0, 0.0))
    pitch = _shirt(step, ruling=40).plates()[0].pitch_mm
    unchoked = _shirt(step, choke=0.0, ruling=40).plates()[0].area
    for cells in (0.5, 0.9, 0.99, 1.0, 1.4):
        job = _shirt(step, choke=cells * pitch, ruling=40)
        eroded = not np.array_equal(job.plates()[0].area, unchoked)
        assert _choke_check(job).ok == eroded, f"check disagrees with erode() at {cells} cells"


def test_a_zero_choke_is_not_reported():
    assert [c for c in _shirt(ht.Constant(0.5), choke=0.0).report().checks if c.kind == "choke"] == []


def test_the_garment_profiles_own_choke_does_not_erode_at_its_own_ruling():
    """plastisol_dark_garment_nominal ships 0.3 mm at 45 lpi: 0.53 of a cell, so tonal edges keep their own."""
    p = ht.PressProfile.load("plastisol_dark_garment_nominal")
    job = p.recipe(ht.Canvas.of(20, 20, dpi=100), [("gold", "#F2C230")],
                   source=ht.Constant(0.5), underbase=True)
    check = _choke_check(job)
    assert not check.ok and "0.53 of a 0.564 mm cell" in check.detail
