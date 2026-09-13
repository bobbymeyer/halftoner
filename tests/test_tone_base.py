"""Tone relative to the base: light inks on dark garments, and the paper case unchanged."""

import numpy as np
import pytest

import halftoner as ht


def _gradient(color, paper, opacity=0.0, tone_range="auto", underbase=None):
    return ht.Recipe(
        canvas=ht.Canvas.of(40, 10, dpi=100),
        inks=ht.InkSet(ht.Ink("ink", color, angle=45, opacity=opacity)),
        screen=ht.Screen(ruling_lpi=40),
        source=ht.Function(lambda x, y: x / 40, kind="tone"),  # image reflectance 0 at left, 1 at right
        substrate=ht.Substrate("s", paper=paper),
        transfer=ht.Transfer(tone_range=tone_range),
        underbase=underbase,
    )


def _area_at(job, t):
    plate = job.plates()[-1]
    X, Y = plate.centers()
    near = (np.abs(X - 40 * t) < 0.8) & (Y > 2) & (Y < 8)
    return float(plate.area[near].mean())


def test_light_ink_on_a_dark_garment_maps_image_white_to_the_solid():
    job = _gradient("#F2C230", "#161618", opacity=0.9)
    tone = job.tone_range_for(job.inks.inks[0])
    assert tone.solid > tone.base and job.transfer.tone_mode(tone) == "range"
    assert _area_at(job, 0.1) < _area_at(job, 0.5) < _area_at(job, 0.9)  # more ink where the image is lighter
    assert _area_at(job, 0.5) == pytest.approx(0.5, abs=0.05)  # n = 1: linear in reflectance across the range
    assert "range (image white = lighter end" in str(job.report())


def test_dark_ink_on_white_paper_is_unchanged():
    job = _gradient("#000000", "#FFFFFF")
    assert job.transfer.tone_mode(job.tone_range_for(job.inks.inks[0])) == "paper"
    for t in (0.2, 0.5, 0.8):
        assert _area_at(job, t) == pytest.approx(float(ht.tone_to_area(t, 0.0)), abs=0.03)


def test_range_mode_spreads_a_light_ink_that_paper_mode_clips():
    paper = _gradient("#F2C230", "#FFFFFF", tone_range="paper")
    spread = _gradient("#F2C230", "#FFFFFF", tone_range="range")
    assert _area_at(paper, 0.2) == 1.0 and _area_at(paper, 0.4) == 1.0  # darker than yellow can go: clipped
    assert 1.0 > _area_at(spread, 0.2) > _area_at(spread, 0.4) > 0.0


def test_an_underbase_raises_a_colors_solid():
    bare = _gradient("#1F3A63", "#161618", opacity=0.8)
    white = ht.Underbase(ink=ht.Ink("white", "#F4F4F0", angle=15, opacity=0.9))
    based = _gradient("#1F3A63", "#161618", opacity=0.8, underbase=white)
    assert based.tone_range_for(based.inks.inks[0]).solid > bare.tone_range_for(bare.inks.inks[0]).solid


def test_tone_range_is_validated_and_saved(tmp_path):
    with pytest.raises(ValueError):
        ht.Transfer(tone_range="nonsense")
    job = _gradient("#F2C230", "#161618", tone_range="range")
    job.source = ht.Gradient(0.0, 1.0, kind="tone")
    job.save(tmp_path / "j.json")
    assert ht.Recipe.load(tmp_path / "j.json").transfer.tone_range == "range"
