"""Closing the loop: print a step wedge, scan it, calibrate the profile, and the next job prints true."""

import pytest

import halftoner as ht
from halftoner.cli import main
from halftoner.color import srgb_to_linear
from halftoner.measure import measure_scan
from halftoner.measure.tone import wedge_boxes
from halftoner.render.raster import composite, save_png

DPI = 800
PRESS_GAIN = 0.2  # what the "real" press does, unknown to the profile


def _press(source, gain=PRESS_GAIN, canvas=(70, 20)):
    """A black-ink press that gains PRESS_GAIN and doesn't compensate: it prints the plate it's given."""
    return ht.Recipe(
        canvas=ht.Canvas.of(*canvas, dpi=DPI),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=45)),
        screen=ht.Screen(ruling_lpi=100),
        source=source,
        substrate=ht.Substrate("press", gain=ht.Curve.gain(gain), min_dot=0.04, max_dot=0.9),
        transfer=ht.Transfer(compensate_gain=False),
    )


@pytest.fixture(scope="module")
def wedge_scan(tmp_path_factory):
    boxes = wedge_boxes(2.0, 5.0, 6.0)
    path = tmp_path_factory.mktemp("wedge") / "wedge.png"
    job = _press(ht.Layered([ht.Masked(ht.Constant(v), ht.Rect(*box)) for box, v in boxes]))
    save_png(composite(job, supersample=2), path, DPI)
    return path


def test_calibration_replaces_only_what_the_wedge_measures(wedge_scan, tmp_path):
    base = ht.PressProfile.load("uncoated_offset_nominal")
    m = measure_scan(wedge_scan, n_inks=1, wedge=(2.0, 5.0, 6.0), patch_ink="ink1")
    cal = m.calibrate_profile(base, "my_press")

    s = cal.substrate
    assert float(s.gain(0.5)) == pytest.approx(float(ht.Curve.gain(PRESS_GAIN)(0.5)), abs=0.02)
    assert (s.min_dot, s.max_dot) == (0.05, 0.9)
    assert s.paper == base.substrate.paper and s.ruling_ceiling_lpi == base.substrate.ruling_ceiling_lpi
    assert cal.ruling_lpi == base.ruling_lpi and cal.press == base.press and cal.angles == base.angles
    assert cal.provenance.startswith("DRAFT") and "wedge.png" in cal.provenance and not cal.measured
    assert len(cal.measurements) == len(base.measurements) + 1
    assert cal.measurements[-1]["calibrated"] == ["substrate.gain", "substrate.min_dot", "substrate.max_dot"]
    assert cal.transfer["yule_nielsen_n"] == 1.0

    cal.save(tmp_path / "my_press.json")
    assert ht.PressProfile.load(tmp_path / "my_press.json").to_dict() == cal.to_dict()


def test_a_calibrated_plate_prints_the_intended_tone(wedge_scan):
    cal = measure_scan(wedge_scan, n_inks=1, wedge=(2.0, 5.0, 6.0), patch_ink="ink1").calibrate_profile(
        ht.PressProfile.load("uncoated_offset_nominal"))

    def printed(plate_area):
        px = composite(_press(ht.Constant(plate_area), canvas=(20, 20)), supersample=2)
        return 1 - float(srgb_to_linear(px / 255.0).mean())  # black ink on white: coverage = 1 - reflectance

    assert printed(0.5) == pytest.approx(0.7, abs=0.03)  # uncalibrated: the press gains a 50% dot to ~70%
    assert printed(float(cal.substrate.gain.inverse()(0.5))) == pytest.approx(0.5, abs=0.03)  # calibrated: true


def test_calibrate_cli_writes_a_profile(wedge_scan, tmp_path, capsys):
    out = tmp_path / "cal.json"
    assert main(["calibrate", "uncoated_offset_nominal", str(wedge_scan), "--wedge", "2,5,6",
                 "--patch-ink", "ink1", "--out", str(out)]) == 0
    assert "printed at a 50% plate dot" in capsys.readouterr().out
    assert ht.PressProfile.load(out).name == "uncoated_offset_nominal_calibrated"


def test_calibration_without_patches_is_refused(wedge_scan):
    m = measure_scan(wedge_scan, n_inks=1)
    with pytest.raises(ValueError, match="step wedge"):
        m.calibrate_profile(ht.PressProfile.load("uncoated_offset_nominal"))
