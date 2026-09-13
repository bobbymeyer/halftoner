"""Vector dot geometry must carry the same ink as the plate says."""

import numpy as np
import pytest

import halftoner as ht
from halftoner.render.geometry import dot_geometry


def _plate(shape, source):
    job = ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=100),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=30)),
        screen=ht.Screen(ruling_lpi=40, shape=shape),
        source=source,
    )
    return job, job.plates()[0]


def _polygon_areas(polygons, keep):
    out = []
    for poly, k in zip(polygons, keep):
        p = poly[k]
        out.append(0.5 * abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
    return np.array(out)


@pytest.mark.parametrize("shape", [ht.Round(), ht.Square(), ht.Elliptical(1.4), ht.Diamond(1.6), ht.Line()])
@pytest.mark.parametrize("tolerance", [None, 0.005])
def test_vector_dots_cover_the_plate_area(shape, tolerance):
    job, plate = _plate(shape, ht.Gradient(0.03, 1.0))
    g = dot_geometry(plate, job.canvas, tolerance_mm=tolerance)
    drawn = np.pi * np.sum(g.circles[:, 2] ** 2) + _polygon_areas(g.polygons, g.keep).sum()
    X, Y = plate.centers()
    pad = plate.pitch_mm
    near = (plate.area > 0) & (X > -pad) & (X < 20 + pad) & (Y > -pad) & (Y < 20 + pad)
    expected = plate.area[near].sum() * plate.pitch_mm**2
    assert drawn == pytest.approx(expected, rel=0.01)


def test_solid_cells_are_exact_squares():
    job, plate = _plate(ht.Elliptical(1.4), ht.Constant(1.0))
    g = dot_geometry(plate, job.canvas, tolerance_mm=0.005)
    assert (g.keep.sum(axis=1) == 4).all()
    assert _polygon_areas(g.polygons, g.keep) == pytest.approx(plate.pitch_mm**2, rel=1e-6)
