import json

import numpy as np
import pytest
from PIL import Image as PILImage

import halftoner as ht


def test_bundled_profiles_load_and_are_labeled():
    names = ht.PressProfile.available()
    assert {"uncoated_offset_nominal", "newsprint_nominal"} <= set(names)
    for n in names:
        p = ht.PressProfile.load(n)
        assert p.substrate.name == n
        assert not p.measured  # nothing bundled is measured yet
        assert ht.Substrate.load(n).paper == p.substrate.paper


def test_profile_round_trips(tmp_path):
    p = ht.PressProfile.load("uncoated_offset_nominal")
    p.save(tmp_path / "p.json")
    assert ht.PressProfile.load(tmp_path / "p.json").to_dict() == p.to_dict()


def test_recipe_from_profile_applies_defaults_and_overrides():
    p = ht.PressProfile.load("newsprint_nominal")
    job = p.recipe(
        canvas=ht.Canvas.of(40, 40, dpi=100),
        inks=[("red", "#D6422B"), ("blue", "#1F3A63", {"density": 1.2})],
        source=ht.Constant(0.3),
        seed=5,
        press={"misregistration": 0.5},
    )
    red, blue = job.inks.inks
    assert (red.angle, blue.angle) == (45, 75)
    assert red.density == 0.9 and blue.density == 1.2
    assert job.screen.ruling_lpi == 65
    assert job.press.misregistration == 0.5 and job.press.slur == 0.04 and job.press.seed == 5
    assert job.substrate.compression == (0.1, 0.85)
    rep = str(job.report())
    assert "newsprint_nominal" in rep and "NOMINAL" in rep


def _full_recipe(tmp_path):
    img = tmp_path / "src.png"
    PILImage.fromarray((np.random.default_rng(0).random((32, 48)) * 255).astype(np.uint8)).save(img)
    photo = ht.Image(img, fit="cover", box=(0, 0, 40, 30))
    inks = ht.InkSet(
        ht.Ink("red", "#D6422B", angle=15, curve=ht.Curve.from_points([(0.5, 0.6)]),
               source=ht.Layered([ht.Masked(photo, ht.Rect(0, 0, 40, 30)),
                                  ht.Masked(ht.Constant(0.25), ht.Polygon(((0, 30), (40, 30), (20, 40))))])),
        ht.Ink("blue", "#1F3A63", angle=45, shape=ht.Diamond(1.3), coverage=0.2),
        overprint={("red", "blue"): "#3A2036"},
    )
    return ht.Recipe(
        canvas=ht.Canvas.of(40, 40, dpi=100),
        inks=inks,
        screen=ht.Screen(ruling_lpi=40, shape=ht.Elliptical(1.4)),
        source=ht.Gradient(0.0, 0.8, (0, 0), (0, 40)),
        substrate=ht.Substrate.load("uncoated_offset_nominal"),
        press=ht.Press(misregistration=0.3, extra_gain=ht.Curve.gain(0.1)),
        transfer=ht.Transfer(yule_nielsen_n=1.5),
        seed=11,
        profile="uncoated_offset_nominal",
    )


def test_recipe_round_trips_through_json(tmp_path):
    job = _full_recipe(tmp_path)
    job.save(tmp_path / "job.json")
    saved = json.loads((tmp_path / "job.json").read_text())
    assert saved["inks"][0]["source"]["sources"][0]["source"]["path"] == "src.png"  # relative to recipe

    back = ht.Recipe.load(tmp_path / "job.json")
    assert back.to_dict(tmp_path) == job.to_dict(tmp_path)
    for a, b in zip(job.plates(), back.plates()):
        assert np.array_equal(a.area, b.area)
        assert np.array_equal(a.printed, b.printed)


def test_function_source_refuses_to_serialize():
    job = ht.Recipe(
        canvas=ht.Canvas.of(10, 10), inks=ht.InkSet(ht.Ink("k", "#000")),
        screen=ht.Screen(40), source=ht.Function(lambda x, y: x * 0),
    )
    with pytest.raises(TypeError):
        job.to_dict()
