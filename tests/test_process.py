"""CMYK process separations: the separation, the plates, TAC, PDF process color, persistence."""

import re

import numpy as np
import pytest
from PIL import Image as PILImage
from pypdf import PdfReader

import halftoner as ht
from halftoner.cli import main
from halftoner.color import srgb_to_linear
from halftoner.policy import ConstraintRefused
from halftoner.process import CmykSeparation, ProcessChannel, process_inks


def _rgb(*srgb):
    return srgb_to_linear(np.array(srgb, dtype=np.float64))


def test_separation_math():
    sep = CmykSeparation(gcr=0.7, black_start=0.2, tac=3.0)
    assert sep.separate(_rgb(1, 1, 1)) == pytest.approx([0, 0, 0, 0])
    c, m, y, k = sep.separate(_rgb(1, 0, 0))
    assert (c, k) == pytest.approx((0, 0)) and (m, y) == pytest.approx((1, 1))
    gray = sep.separate(_rgb(0.5, 0.5, 0.5))
    assert gray[3] == pytest.approx(0.7 * (0.3 / 0.8) * 0.5)  # black starts above 20% gray
    assert gray[0] == pytest.approx((0.5 - gray[3]) / (1 - gray[3]))
    black = sep.separate(_rgb(0, 0, 0))
    assert black[3] == pytest.approx(0.7) and black.sum() == pytest.approx(3.0)  # rich black held to 300%
    assert CmykSeparation(tac=2.4).separate(_rgb(0, 0, 0)).sum() == pytest.approx(2.4)
    assert sep.separate(_rgb(0.9, 0.9, 0.9))[3] == 0.0  # light grays carry no black


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    """Quadrants: red, mid gray, black, white."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)  # 10 px/mm on the 40 mm canvas: sampling ratio well over 1
    img[:200, :200] = (255, 0, 0)
    img[:200, 200:] = (128, 128, 128)
    img[200:, :200] = (0, 0, 0)
    img[200:, 200:] = (255, 255, 255)
    path = tmp_path_factory.mktemp("photo") / "quadrants.png"
    PILImage.fromarray(img).save(path)
    return path


def _job(photo, tac=None, separation=None):
    return ht.Recipe(
        canvas=ht.Canvas.of(40, 40, dpi=100),
        inks=process_inks(photo, separation),
        screen=ht.Screen(ruling_lpi=60),
        substrate=ht.Substrate("stock", tac=tac, ruling_ceiling_lpi=150),
    )


def _mean_in(plate, x0, x1, y0, y1):
    X, Y = plate.centers()
    sel = (X > x0) & (X < x1) & (Y > y0) & (Y < y1)
    return float(plate.area[sel].mean())


def test_process_plates_in_kcmy_order_at_classic_angles(photo):
    job = _job(photo)
    names = [p.ink.name for p in job.plates()]
    assert names == ["black", "cyan", "magenta", "yellow"]
    assert [job.angle_for(i) for i in job.inks] == [45, 15, 75, 0]
    angles = [c for c in job.report().checks if c.kind == "angle"]
    assert len(angles) == 6 and all(c.ok for c in angles)  # every KCMY pair, all at one ruling
    black, cyan, magenta, yellow = job.plates()
    red = (2, 18, 2, 18)  # the red quadrant, inset from its edges
    assert _mean_in(magenta, *red) == pytest.approx(1.0) and _mean_in(yellow, *red) == pytest.approx(1.0)
    assert _mean_in(cyan, *red) == pytest.approx(0.0) and _mean_in(black, *red) == pytest.approx(0.0)
    white = (22, 38, 22, 38)
    assert all(_mean_in(p, *white) == pytest.approx(0.0) for p in job.plates())
    assert _mean_in(black, 2, 18, 22, 38) == pytest.approx(0.7, abs=0.01)  # the black quadrant


def test_tac_is_checked_and_refused_for_press(photo, tmp_path):
    loose = _job(photo, tac=2.6, separation=CmykSeparation(tac=4.0))
    check = next(c for c in loose.report().checks if c.kind == "tac")
    assert not check.ok and "260%" in check.detail
    with pytest.raises(ConstraintRefused):
        loose.render("pdf", tmp_path)
    held = _job(photo, tac=2.6, separation=CmykSeparation(tac=2.6))
    assert next(c for c in held.report().checks if c.kind == "tac").ok


def test_pdf_writes_process_inks_as_device_cmyk(photo, tmp_path):
    job = _job(photo)
    spot = ht.Ink("gold", "#C9A227", angle=30, source=ht.Masked(ht.Constant(1.0), ht.Rect(5, 5, 5, 5)))
    job.inks = ht.InkSet(*job.inks.inks, spot)
    path = tmp_path / "cmyk.pdf"
    job.render_pdf(path, output_condition="FOGRA39", mode="bitmap", bitmap_dpi=300)
    page = PdfReader(path).pages[0]
    spaces = page["/Resources"]["/ColorSpace"]
    seps = {str(spaces[key][1]) for key in spaces}
    assert seps == {"/gold", "/All"}  # only the spot ink is a Separation
    content = page.get_contents().get_data().decode("ascii")
    for fill in ("0 0 0 1 k", "1 0 0 0 k", "0 1 0 0 k", "0 0 1 0 k"):
        assert re.search(rf"q {fill} [-\d. ]+ cm /Im\d Do", content), fill
    assert re.search(r"/CS\d+ cs 1 scn [-\d. ]+ cm /Im4 Do", content)


def test_process_recipe_round_trips_and_scales(photo, tmp_path):
    job = _job(photo, tac=3.0)
    job.save(tmp_path / "cmyk.json")
    back = ht.Recipe.load(tmp_path / "cmyk.json")
    assert back.to_dict(tmp_path) == job.to_dict(tmp_path)
    for a, b in zip(job.plates(), back.plates()):
        assert np.array_equal(a.area, b.area)
    big = job.at_size(80, 80)
    assert isinstance(big.inks.inks[0].source, ProcessChannel)
    assert big.report().inks[0].mean_plate == pytest.approx(job.report().inks[0].mean_plate, abs=0.02)


def test_cli_new_process(photo, tmp_path):
    out = tmp_path / "job.json"
    assert main(["new", str(out), "--profile", "uncoated_offset_nominal", "--image", str(photo),
                 "--size", "40x40", "--dpi", "100", "--process"]) == 0
    job = ht.Recipe.load(out)
    assert [i.process for i in job.inks] == ["k", "c", "m", "y"]
    assert job.inks.inks[0].source.separation.tac == job.substrate.tac
