"""Measure against renders whose every parameter is known."""

import numpy as np
import pytest

import halftoner as ht
from halftoner.cli import main
from halftoner.color import hex_to_linear, linear_to_hex
from halftoner.measure import measure_scan, phase_correlate
from halftoner.render.raster import composite, save_png

DPI = 800
RULING = 100


def _blobs(x, y):
    rng = np.random.default_rng(1)
    v = np.ones_like(x)
    for cx, cy, r, d in zip(rng.uniform(0, 40, 14), rng.uniform(0, 40, 14), rng.uniform(2, 7, 14), rng.uniform(0.3, 0.7, 14)):
        v = v - d * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))
    return np.clip(v, 0.15, 1.0)


def _art(x, y):
    """Soft blobs plus hard-edged shapes spread over the whole sheet.

    Registration is read off shared edges, as on a real sheet; every tile needs some.
    """
    v = 0.4 + 0.6 * _blobs(x, y)
    rng = np.random.default_rng(4)
    shapes = zip(rng.uniform(-2, 38, 36), rng.uniform(-2, 38, 36), rng.uniform(2, 8, 36),
                 rng.uniform(2, 8, 36), rng.uniform(0.2, 0.5, 36))
    for x0, y0, w, h, d in shapes:
        v = np.where((x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h), v - d, v)
    return np.clip(v, 0.1, 1.0)


def _angle_diff(a, b):
    return abs((a - b + 45) % 90 - 45)


@pytest.fixture(scope="module")
def duotone(tmp_path_factory):
    photo = ht.Function(_art, kind="tone")
    job = ht.Recipe(
        canvas=ht.Canvas.of(40, 40, dpi=DPI),
        inks=ht.InkSet(ht.Ink("blue", "#1F3A63", angle=45), ht.Ink("red", "#D6422B", angle=15)),
        screen=ht.Screen(ruling_lpi=RULING),
        source=photo,
        substrate=ht.Substrate("t", paper="#F2EDE0"),
        press=ht.Press(misregistration=0.4, drift=0.0, seed=2),  # blue is the key plate
    )
    path = tmp_path_factory.mktemp("scan") / "duotone.png"
    save_png(composite(job, supersample=2), path, DPI)
    return job, path


def test_phase_correlate_sign():
    rng = np.random.default_rng(0)
    a = rng.random((128, 128))
    b = np.roll(np.roll(a, 5, axis=0), -3, axis=1)
    dy, dx, _ = phase_correlate(a, b)
    assert (dy, dx) == pytest.approx((5, -3), abs=0.05)


def test_recovers_paper_inks_ruling_and_angles(duotone):
    job, path = duotone
    m = measure_scan(path, n_inks=2)
    assert m.dpi == DPI  # read from the file
    assert np.abs(m.model.paper - hex_to_linear("#F2EDE0")).max() < 0.02

    blue, red = job.inks.inks
    expected = {"ink1": blue, "ink2": red}  # darkest first
    for i, name in enumerate(m.model.names):
        ink = expected[name]
        assert np.abs(hex_to_linear(m.model.ink_hex(i)) - ink.transmittance).max() < 0.03, (
            name, m.model.ink_hex(i), linear_to_hex(ink.transmittance))
        s = m.screens[i]
        assert s.ruling_lpi == pytest.approx(RULING, rel=0.01)
        assert _angle_diff(s.angle, ink.angle) < 0.5


def test_recovers_misregistration(duotone):
    job, path = duotone
    m = measure_scan(path, n_inks=2, names=["blue", "red"])
    field = job.press.offsets(job.inks.inks, job.canvas)["red"]
    dx, dy = (float(v[0]) for v in field(np.array([20.0]), np.array([20.0])))
    reg = m.registration["red"]
    assert reg is not None
    assert (reg.dx_mm, reg.dy_mm) == pytest.approx((dx, dy), abs=0.05)
    assert len(reg.tiles) >= 5
    assert reg.noise_mm < 0.1
    assert not reg.walk_resolved  # the press in this render doesn't walk, so no walk may be claimed


def test_survives_scanner_blur_and_noise(tmp_path):
    """A realistic scan: 12 px per cell, optics blur in linear light, sensor noise.

    Edge pixels become a continuum rather than the discrete blends of a supersampled render.
    """
    from PIL import Image as PILImage

    from halftoner.color import linear_to_srgb
    from halftoner.measure.common import gaussian_blur, to_linear

    dpi = 1200
    job = ht.Recipe(
        canvas=ht.Canvas.of(30, 30, dpi=dpi),
        inks=ht.InkSet(ht.Ink("blue", "#1F3A63", angle=45), ht.Ink("red", "#D6422B", angle=15)),
        screen=ht.Screen(ruling_lpi=RULING),
        source=ht.Function(_art, kind="tone"),
        substrate=ht.Substrate("t", paper="#F2EDE0"),
        press=ht.Press(misregistration=0.4, drift=0.0, seed=2),
    )
    lin = to_linear(composite(job, supersample=2))
    blurred = np.stack([gaussian_blur(lin[..., c], 0.8) for c in range(3)], axis=-1)
    srgb = linear_to_srgb(np.clip(blurred, 0, 1)) * 255 + np.random.default_rng(3).normal(0, 1.5, lin.shape)
    scan = tmp_path / "scanned.png"
    PILImage.fromarray(np.clip(np.round(srgb), 0, 255).astype(np.uint8)).save(scan, dpi=(dpi, dpi))

    m = measure_scan(scan, n_inks=2, names=["blue", "red"])
    blue, red = job.inks.inks
    for i, ink in enumerate((blue, red)):
        assert np.abs(hex_to_linear(m.model.ink_hex(i)) - ink.transmittance).max() < 0.05, (ink.name, m.model.ink_hex(i))
        assert m.screens[i].ruling_lpi == pytest.approx(RULING, rel=0.01)
        assert _angle_diff(m.screens[i].angle, ink.angle) < 0.5
    assert not [w for w in m.warnings if "not observed" in w]
    field = job.press.offsets(job.inks.inks, job.canvas)["red"]
    dx, dy = (float(v[0]) for v in field(np.array([15.0]), np.array([15.0])))
    reg = m.registration["red"]
    assert (reg.dx_mm, reg.dy_mm) == pytest.approx((dx, dy), abs=0.07)


class _WalkingPress(ht.Press):
    """Red plate offset by (0.2, -0.1) mm at the sheet center, walking linearly toward the edges."""

    def offsets(self, inks, canvas):
        zero = lambda x, y: (np.zeros_like(x), np.zeros_like(y))  # noqa: E731
        walk = lambda x, y: (0.2 + 0.01 * (x - 20), -0.1 + 0.008 * (y - 20))  # noqa: E731
        return {"blue": zero, "red": walk}


def test_resolves_a_real_walk(tmp_path):
    job = ht.Recipe(
        canvas=ht.Canvas.of(40, 40, dpi=DPI),
        inks=ht.InkSet(ht.Ink("blue", "#1F3A63", angle=45), ht.Ink("red", "#D6422B", angle=15)),
        screen=ht.Screen(ruling_lpi=RULING),
        source=ht.Function(_art, kind="tone"),
        substrate=ht.Substrate("t", paper="#F2EDE0"),
        press=_WalkingPress(),
    )
    path = tmp_path / "walk.png"
    save_png(composite(job, supersample=2), path, DPI)
    reg = measure_scan(path, n_inks=2, names=["blue", "red"]).registration["red"]
    assert (reg.dx_mm, reg.dy_mm) == pytest.approx((0.2, -0.1), abs=0.06)
    assert reg.walk_resolved
    assert reg.drift_mm == pytest.approx(np.hypot(0.2, 0.16), abs=0.08)  # corner departure of the linear walk


def test_recovers_gain_and_tone_limits_from_step_wedge(tmp_path):
    from halftoner.measure.tone import wedge_boxes

    boxes = wedge_boxes(2.0, 5.0, 6.0)
    job = ht.Recipe(
        canvas=ht.Canvas.of(70, 20, dpi=DPI),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=45)),
        screen=ht.Screen(ruling_lpi=RULING),
        source=ht.Layered([ht.Masked(ht.Constant(v), ht.Rect(*box)) for box, v in boxes]),
        substrate=ht.Substrate("t", gain=ht.Curve.gain(0.2), min_dot=0.04, max_dot=0.9),
        transfer=ht.Transfer(compensate_gain=False),
    )
    path = tmp_path / "wedge.png"
    save_png(composite(job, supersample=2), path, DPI)

    m = measure_scan(path, n_inks=1, wedge=(2.0, 5.0, 6.0), patch_ink="ink1")
    gain = ht.Curve.gain(0.2)
    for r in m.patches:
        if r.nominal in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            assert r.coverage == pytest.approx(float(gain(r.nominal)), abs=0.02), r
    profile = m.draft_profile("wedge_test")
    assert profile["substrate"]["min_dot"] == 0.05
    assert profile["substrate"]["max_dot"] == 0.9
    assert m.screens[0].ruling_lpi == pytest.approx(RULING, rel=0.01)


def test_draft_profile_loads_and_is_not_measured(duotone, tmp_path, capsys):
    _, path = duotone
    out = tmp_path / "draft.json"
    assert main(["measure", str(path), "--inks", "2", "--names", "blue,red", "--profile-out", str(out)]) == 0
    assert "blue" in capsys.readouterr().out
    p = ht.PressProfile.load(out)
    assert not p.measured and p.provenance.startswith("DRAFT")
    assert p.ruling_lpi == pytest.approx(RULING, rel=0.01)
    assert p.press["misregistration"] > 0.1
