"""Print-on-demand: render at any size, a file per band, transparent backgrounds."""

import json

import numpy as np
import pytest
from PIL import Image as PILImage

import halftoner as ht
from halftoner.cli import main
from halftoner.color import hex_to_linear, srgb_to_linear
from halftoner.pod import SizeBand, bands_for, load_bands
from halftoner.render.raster import composite


def _design(**press):
    red = ht.Ink("red", "#D6422B", angle=15, source=ht.Masked(ht.Constant(1.0), ht.Rect(10, 10, 20, 20)))
    blue = ht.Ink("blue", "#1F3A63", angle=45, source=ht.Gradient(0.1, 0.8, (0, 0), (60, 0)))
    return ht.Recipe(
        canvas=ht.Canvas.of(60, 40, dpi=100, bleed=3),
        inks=ht.InkSet(red, blue),
        screen=ht.Screen(ruling_lpi=60, origin=(1.0, 2.0), grid=ht.ModuleGrid(20.0, origin=(5.0, 5.0))),
        substrate=ht.Substrate("stock", paper="#E6DFCC"),
        press=ht.Press(misregistration=0.3, slur=0.05, trap_gap=0.1, **press),
    )


def test_at_size_scales_the_art_not_the_press():
    job = _design()
    big = job.at_size(120, 80, dpi=50)
    assert (big.canvas.width_mm, big.canvas.height_mm, big.canvas.dpi, big.canvas.bleed_mm) == (120, 80, 50, 3)
    assert big.inks.inks[0].source.region == ht.Rect(20, 20, 40, 40)
    assert big.inks.inks[1].source.p1 == (120, 0)
    assert big.screen.grid.repeat_mm == 40 and big.screen.grid.origin == (10, 10) and big.screen.origin == (2, 4)
    assert (big.press.misregistration, big.press.slur, big.press.trap_gap) == (0.3, 0.05, 0.1)
    assert big.screen.ruling_lpi == 60
    assert job.canvas.width_mm == 60  # the original is untouched
    with pytest.raises(ValueError, match="aspect"):
        job.at_size(120, 120)


def test_coverage_holds_across_sizes():
    job = _design()
    big = job.at_size(180, 120)
    for small_plate, big_plate in zip(job.report().inks, big.report().inks):
        assert big_plate.mean_plate == pytest.approx(small_plate.mean_plate, abs=0.02)


def test_bands_match_aspect_in_either_orientation():
    names = lambda w, h: [(b.name, b.family) for b in bands_for(ht.Canvas.of(w, h))]  # noqa: E731
    portrait = bands_for(ht.Canvas.of(200, 300))
    assert [b.family for b in portrait] == ["2:3"] * 3 and all(b.width_mm < b.height_mm for b in portrait)
    assert all(b.width_mm > b.height_mm for b in bands_for(ht.Canvas.of(300, 200)))
    assert {f for _, f in names(100, 100)} == {"1:1"}
    assert names(100, 37) == []
    assert all(b.max_lpi == b.dpi / 5 for b in load_bands())


BANDS = [SizeBand("S", 40, 60, 100, 30, "2:3"), SizeBand("M", 80, 120, 100, 30, "2:3")]


def _portrait():
    return ht.Recipe(
        canvas=ht.Canvas.of(20, 30, dpi=100),
        inks=ht.InkSet(ht.Ink("k", "#1F3A63", angle=45, source=ht.Masked(ht.Constant(1.0), ht.Rect(5, 5, 10, 10)))),
        screen=ht.Screen(ruling_lpi=60),
        substrate=ht.Substrate("stock", paper="#E6DFCC"),
    )


def test_a_file_per_band_capped_and_sized(tmp_path):
    out = _portrait().render("pod", tmp_path, stem="job", size="all", bands=BANDS, supersample=2)
    assert [p.name for p in out.paths] == ["job_pod_S.png", "job_pod_M.png"]
    sizes = [PILImage.open(p).size for p in out.paths]
    assert sizes == [(round(40 / 25.4 * 100), round(60 / 25.4 * 100)), (round(80 / 25.4 * 100), round(120 / 25.4 * 100))]
    assert out.capped["S/k"] == (60, 30) and out.capped["M/k"] == (60, 30)
    one = _portrait().render("pod", tmp_path, stem="one", size="M", bands=BANDS, supersample=2)
    assert [p.name for p in one.paths] == ["one_pod_M.png"]
    with pytest.raises(ValueError, match="no band named"):
        _portrait().render("pod", tmp_path, size="XL", bands=BANDS)


def test_hard_and_soft_alpha(tmp_path):
    job = _portrait()
    hard = composite(job, supersample=4, alpha="hard")
    soft = composite(job, supersample=4, alpha="soft")
    assert hard.shape[-1] == 4 and set(np.unique(hard[..., 3])) <= {0, 255}
    assert ((soft[..., 3] > 0) & (soft[..., 3] < 255)).any()  # partial pixels at dot and region edges
    assert (hard[:18, :, 3] == 0).all()  # above the panel (y < 5 mm at ~3.9 px/mm): nothing printed, transparent
    inside = hard[30:55, 30:55]
    assert (inside[..., 3] == 255).all()
    # Inks are composited over white, not the stock: the substrate isn't baked into a transparent file.
    assert np.abs(srgb_to_linear(inside[..., :3] / 255.0).mean(axis=(0, 1)) - hex_to_linear("#1F3A63")).max() < 0.01
    opaque = composite(job, supersample=4)
    assert opaque.shape[-1] == 3


def test_cli_pod_sizes_with_a_bands_file(tmp_path):
    bands = tmp_path / "bands.json"
    bands.write_text(json.dumps({"families": {"2:3": [{"name": "S", "size_mm": [40, 60], "dpi": 100, "max_lpi": 30}]}}))
    recipe = tmp_path / "job.json"
    _portrait().save(recipe)
    assert main(["render", str(recipe), "--target", "pod", "--size", "all", "--bands", str(bands),
                 "--out", str(tmp_path), "--supersample", "2"]) == 0
    img = PILImage.open(tmp_path / "job_pod_S.png")
    assert img.mode == "RGBA"
