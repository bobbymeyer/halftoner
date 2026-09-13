"""Grid-locked screens: every layout module carries the same dots."""

import math

import numpy as np
import pytest

import halftoner as ht
from halftoner.policy import apply_policy
from halftoner.render.bitmap import hit_rows

GRID = ht.ModuleGrid(20.0, origin=(5.0, 5.0))


def _job(angle=15, lpi=85, grid=GRID, ceiling=150, shape=None):
    return ht.Recipe(
        canvas=ht.Canvas.of(60, 40, dpi=254),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=angle)),
        screen=ht.Screen(ruling_lpi=lpi, grid=grid, shape=shape or ht.Round()),
        source=ht.Gradient(0.2, 0.7),
        substrate=ht.Substrate("t", ruling_ceiling_lpi=ceiling),
    )


def test_lock_snaps_to_a_rational_angle_and_whole_repeats():
    lock = GRID.lock(85, 15)
    assert (lock.p, lock.q) == (4, 1) and lock.angle_deg == pytest.approx(math.degrees(math.atan(0.25)))
    assert lock.periods * lock.period_mm == pytest.approx(20.0)
    assert abs(lock.ruling_lpi - 85) / 85 < 0.06
    assert GRID.lock(85, 105).angle_deg == pytest.approx(lock.angle_deg + 90)  # quadrant kept for elliptical dots
    assert GRID.lock(85, 45).angle_deg == 45.0 and GRID.lock(85, 0).angle_deg == 0.0
    assert GRID.lock(85, 15, at_most=True).ruling_lpi <= 85


@pytest.mark.parametrize("angle", [0, 15, 45, 75, 22.5])
def test_the_lattice_repeats_exactly_on_the_grid(angle):
    job = _job(angle=angle)
    plate = job.plates()[0]
    for dx, dy in ((20.0, 0.0), (0.0, 20.0)):
        s, t = plate.to_screen(np.array([5.0 + dx]), np.array([5.0 + dy]))
        s0, t0 = plate.to_screen(np.array([5.0]), np.array([5.0]))
        steps = np.array([(s - s0)[0], (t - t0)[0]]) / plate.pitch_mm
        assert np.allclose(steps, np.round(steps), atol=1e-9)


def test_identical_dots_in_every_module_of_a_flat_tint():
    job = _job()
    job.source = ht.Constant(0.4)
    plate = job.plates()[0]
    (_, _, _, a), = hit_rows(plate, 5.0, 5.0, 200, 200, 254)  # one 20 mm module at 10 px/mm
    (_, _, _, b), = hit_rows(plate, 25.0, 5.0, 200, 200, 254)  # the next module over
    assert a.mean() > 0.2
    assert np.mean(a != b) < 1e-3


def test_report_shows_the_lock_and_flags_unlocked_rows():
    job = _job(grid=ht.ModuleGrid(20.0, 23.0, origin=(5.0, 5.0)))
    report = job.report()
    assert "grid lock" in str(report) and "tan 1/4" in str(report)
    rows = next(c for c in report.checks if c.kind == "grid")
    assert not rows.ok
    rows = [c for c in _job(grid=ht.ModuleGrid(20.0, 40.0, origin=(5.0, 5.0))).report().checks if c.kind == "grid"]
    assert len(rows) == 1 and rows[0].ok  # the one ink; a 40 mm repeat is whole periods


def test_capping_stays_on_the_grid_and_under_the_ceiling():
    capped, outcome = apply_policy(_job(lpi=85, ceiling=60), "pod")
    ink = capped.inks.inks[0]
    assert capped.ruling_for(ink) <= 60 and outcome.capped["k"][1] <= 60
    lock = capped.grid_lock(ink)
    assert lock.periods * lock.period_mm == pytest.approx(20.0)


def test_grid_round_trips(tmp_path):
    job = _job(grid=ht.ModuleGrid(20.0, 30.0, origin=(5.0, 5.0), max_ratio=4))
    job.save(tmp_path / "g.json")
    back = ht.Recipe.load(tmp_path / "g.json")
    assert back.screen.grid == job.screen.grid
    assert np.array_equal(back.plates()[0].area, job.plates()[0].area)
