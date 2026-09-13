"""Density variance, full-resolution region cuts, and trap gaps."""

import numpy as np
import pytest

import halftoner as ht
from halftoner.color import hex_to_linear, srgb_to_linear
from halftoner.render.raster import composite

DPI = 254  # 10 px/mm, so pixel columns are 0.1 mm


def _solid(**press):
    return ht.Recipe(
        canvas=ht.Canvas.of(60, 40, dpi=DPI),
        inks=ht.InkSet(ht.Ink("k", "#3050A0", angle=45)),
        screen=ht.Screen(ruling_lpi=60),
        source=ht.Constant(1.0),
        substrate=ht.Substrate("t", paper="#FFFFFF"),
        press=ht.Press(**press),
    )


def _film_thickness(px, ink_hex):
    """Relative film thickness of a solid on white: log(reflectance) / log(swatch transmittance)."""
    lin = srgb_to_linear(px / 255.0)
    return (np.log10(np.clip(lin, 1e-6, None)) / np.log10(hex_to_linear(ink_hex))).mean(axis=-1)


def test_density_variance_is_bounded_low_frequency_and_seeded():
    flat = composite(_solid(), supersample=1)
    assert np.ptp(flat.reshape(-1, 3), axis=0).max() <= 1

    a = composite(_solid(density_variance=0.1, seed=4), supersample=1)
    assert np.array_equal(a, composite(_solid(density_variance=0.1, seed=4), supersample=1))
    assert not np.array_equal(a, composite(_solid(density_variance=0.1, seed=5), supersample=1))

    film = _film_thickness(a, "#3050A0")
    assert film.min() >= 0.88 and film.max() <= 1.12  # +-10% plus 8-bit rounding
    assert film.max() - film.min() > 0.05  # it does vary across the sheet
    assert np.abs(np.diff(film.mean(axis=0))).max() < 0.01  # and it walks, it doesn't speckle
    assert np.array_equal(composite(_solid(density_variance=0.1, seed=4), supersample=1, artifacts=False), flat)


def test_region_edges_are_cut_at_full_resolution():
    job = ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=DPI),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=45)),
        screen=ht.Screen(ruling_lpi=40),
        source=ht.Masked(ht.Constant(0.5), ht.Rect(0, 0, 10, 20)),
        substrate=ht.Substrate("t", paper="#FFFFFF"),
    )
    px = composite(job, supersample=4)
    assert (px[:, 100:] == 255).all()  # nothing past the cut at x = 10 mm
    # The tint reaches the cut: edge dots are sliced, not dropped.
    assert srgb_to_linear(px[:, 95:100] / 255.0).mean() == pytest.approx(0.5, abs=0.08)


def test_layer_edge_cells_do_not_leak_into_an_overlapping_layer():
    """A solid panel over a larger tint on the same plate: below the panel edge is tint, not stair-stepped solid."""
    job = ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=DPI),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=45)),
        screen=ht.Screen(ruling_lpi=40),
        source=ht.Layered([
            ht.Masked(ht.Constant(1.0), ht.Rect(0, 0, 20, 10)),
            ht.Masked(ht.Constant(0.3), ht.Rect(0, 0, 20, 20)),
        ]),
        substrate=ht.Substrate("t", paper="#FFFFFF"),
    )
    ink = 1 - srgb_to_linear(composite(job, supersample=4) / 255.0).mean(axis=-1)
    assert ink[95:100].mean() > 0.98  # the panel is solid right to its edge
    just_below, well_below = ink[100:104, 5:-5].mean(), ink[150:154, 5:-5].mean()
    assert just_below == pytest.approx(well_below, abs=0.06)  # tint, with no solid cells poking through


def _abutting(trap_gap=0.0, press=None):
    red = ht.Ink("red", "#D6422B", angle=15, source=ht.Masked(ht.Constant(1.0), ht.Rect(0, 0, 10, 20)))
    blue = ht.Ink("blue", "#1F3A63", angle=75, source=ht.Masked(ht.Constant(1.0), ht.Rect(10, 0, 10, 20)))
    return ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=DPI),
        inks=ht.InkSet(red, blue),
        screen=ht.Screen(ruling_lpi=40),
        substrate=ht.Substrate("t", paper="#FFFFFF"),
        press=press or ht.Press(trap_gap=trap_gap),
    )


def test_trap_gap_opens_paper_only_where_regions_meet():
    butt = composite(_abutting(0.0), supersample=4)
    gap = composite(_abutting(0.3), supersample=4)
    assert not (butt[:, 99:101] == 255).all(axis=-1).any()  # butted: no paper at the seam
    assert (gap[:, 99:101] == 255).all()  # 0.3 mm gap: 9.9-10.1 mm is paper
    assert np.array_equal(gap[5:-5, 90], butt[5:-5, 90]) and np.array_equal(gap[5:-5, 110], butt[5:-5, 110])
    assert np.array_equal(composite(_abutting(0.3), supersample=4, artifacts=False), butt)


class _ShiftBlue(ht.Press):
    def offsets(self, inks, canvas):
        return {
            "red": lambda x, y: (np.zeros_like(x), np.zeros_like(y)),
            "blue": lambda x, y: (np.full_like(x, 0.3), np.zeros_like(y)),
        }


def test_misregistration_moves_region_cuts_with_the_plate():
    px = composite(_abutting(press=_ShiftBlue()), supersample=4)
    assert (px[:, 100:103] == 255).all()  # blue walked 0.3 mm right: paper opens on its side of the seam
    assert not (px[:, 98] == 255).all(axis=-1).any()  # red still reaches its edge


def test_svg_clips_masked_plates(tmp_path):
    _abutting().render_svg(tmp_path / "t.svg")
    text = (tmp_path / "t.svg").read_text()
    assert text.count("<clipPath") == 2
    assert 'clip-path="url(#clip-red)"' in text and 'clip-path="url(#clip-blue)"' in text
    assert text.count("<path") == 2
