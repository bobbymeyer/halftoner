import numpy as np
import pytest
from PIL import Image as PILImage

import halftoner as ht
from halftoner.color import hex_to_linear, srgb_to_linear
from halftoner.render.raster import composite


@pytest.mark.parametrize("shape", [ht.Round(), ht.Square(), ht.Elliptical(1.6), ht.Diamond(), ht.Line()])
@pytest.mark.parametrize("area", [0.05, 0.3, 0.5, 0.8, 0.95])
def test_threshold_inks_requested_area(shape, area):
    g = (np.arange(700) + 0.5) / 700 - 0.5
    u, v = np.meshgrid(g, g)
    inked = (shape.spot(u, v) <= shape.threshold(area)).mean()
    assert inked == pytest.approx(area, abs=0.01)


def test_join_points():
    assert ht.Round().join_points() == [pytest.approx(0.785, abs=0.002)]
    assert ht.Square().join_points() == [pytest.approx(0.5, abs=0.002)]
    assert len(ht.Elliptical(1.6).join_points()) == 2


def test_murray_davies_and_yule_nielsen_round_trip():
    assert ht.tone_to_area(0.5, 0.0, n=1) == pytest.approx(0.5)
    a = ht.tone_to_area(np.linspace(0, 1, 11), 0.2, n=1.7)
    assert ht.area_to_tone(a, 0.2, n=1.7) == pytest.approx(np.clip(np.linspace(0, 1, 11), 0.2, 1))


def test_gain_curve_and_inverse():
    g = ht.Curve.gain(0.15)
    assert g(0.5) == pytest.approx(0.65)
    assert g.inverse()(g(np.array([0.1, 0.5, 0.9]))) == pytest.approx([0.1, 0.5, 0.9], abs=1e-3)


def test_overprint_override_propagates_to_larger_sets():
    inks = ht.InkSet(
        ht.Ink("red", "#D6422B"), ht.Ink("blue", "#1F3A63"), ht.Ink("yellow", "#F2C230"),
        overprint={("red", "blue"): "#3A2036"},
    )
    prim = inks.primaries("#FFFFFF")
    assert prim[0b011] == pytest.approx(hex_to_linear("#3A2036"))
    assert prim[0b111] == pytest.approx(hex_to_linear("#3A2036") * inks.inks[2].transmittance)
    assert prim[0b101] == pytest.approx(inks.inks[0].transmittance * inks.inks[2].transmittance)


def _flat_tint(value, shape=None, **press):
    return ht.Recipe(
        canvas=ht.Canvas.of(20, 20, dpi=150),
        inks=ht.InkSet(ht.Ink("black", "#000000", angle=45)),
        screen=ht.Screen(ruling_lpi=40, shape=shape or ht.Round()),
        source=ht.Constant(value),
        press=ht.Press(**press),
    )


@pytest.mark.parametrize("shape", [ht.Round(), ht.Square(), ht.Elliptical()])
def test_constant_source_is_a_flat_tint_with_correct_reflectance(shape):
    job = _flat_tint(0.3, shape)
    assert job.report().inks[0].mean_plate == pytest.approx(0.3)
    px = composite(job, supersample=4)
    reflectance = srgb_to_linear(px / 255.0).mean()
    assert reflectance == pytest.approx(0.7, abs=0.01)


def test_press_is_seeded_and_key_plate_stays_put():
    canvas = ht.Canvas.of(100, 100)
    inks = ht.InkSet(ht.Ink("a", "#000"), ht.Ink("b", "#f00"), ht.Ink("c", "#00f"))
    x = np.array([10.0, 50.0]); y = np.array([10.0, 90.0])
    f1 = ht.Press(misregistration=0.4, seed=3).offsets(inks.inks, canvas)
    f2 = ht.Press(misregistration=0.4, seed=3).offsets(inks.inks, canvas)
    f3 = ht.Press(misregistration=0.4, seed=4).offsets(inks.inks, canvas)
    assert np.all(f1["a"](x, y)[0] == 0)
    assert np.allclose(f1["b"](x, y), f2["b"](x, y))
    assert not np.allclose(f1["b"](x, y), f3["b"](x, y))


def test_masked_region_leaves_no_ink_outside():
    job = _flat_tint(0.0)
    job.inks.inks[0].source = ht.Masked(ht.Constant(0.5), ht.Rect(0, 0, 10, 20))
    plate = job.plates()[0]
    X, Y = plate.centers()
    assert np.all(plate.area[X > 10.5] == 0)
    inside = (X > 0.5) & (X < 9.5) & (Y > 0.5) & (Y < 19.5)
    assert np.all(plate.area[inside] == pytest.approx(0.5))


def test_substrate_limits_drop_and_snap():
    sub = ht.Substrate("t", min_dot=0.1, max_dot=0.9)
    ink = ht.Ink("k", "#000")
    a = ht.Transfer(compensate_gain=False).plate_area(np.array([0.05, 0.5, 0.95]), "area", ink, sub)
    assert list(a) == [0.0, 0.5, 1.0]


def test_image_sampling_ratio_and_report(tmp_path):
    img = tmp_path / "g.png"
    PILImage.fromarray(np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))).save(img)
    job = ht.Recipe(
        canvas=ht.Canvas.of(64, 64, dpi=100),
        inks=ht.InkSet(ht.Ink("k", "#000", angle=45), ht.Ink("r", "#c00", angle=50)),
        screen=ht.Screen(ruling_lpi=25.4),  # 1 mm cells over 1 mm source px
        source=ht.Image(img),
    )
    rep = job.report()
    assert rep.inks[0].sampling_ratio == pytest.approx(1.0)
    sep = next(c for c in rep.checks if "angle" in c.name)
    assert not sep.ok
    assert "PAST" in str(rep)


def test_svg_has_one_path_per_ink(tmp_path):
    job = _flat_tint(0.4)
    counts = job.render_svg(tmp_path / "t.svg")
    text = (tmp_path / "t.svg").read_text()
    assert text.count("<path") == 1
    assert counts["black"] > 0


@pytest.mark.parametrize("target", [0.12, 0.35, 0.6])
def test_coverage_target_is_hit_through_gain(target):
    job = ht.Recipe(
        canvas=ht.Canvas.of(60, 40, dpi=100),
        inks=ht.InkSet(ht.Ink("k", "#000", coverage=target)),
        screen=ht.Screen(ruling_lpi=40),
        source=ht.Function(lambda x, y: 0.15 + 0.8 * x / 60, kind="tone"),
        substrate=ht.Substrate.load("newsprint_nominal"),
        press=ht.Press(extra_gain=0.1),
    )
    rep = job.report()
    assert rep.inks[0].mean_printed == pytest.approx(target, abs=0.005)
    assert next(c for c in rep.checks if c.kind == "coverage").ok


def test_a_rotated_phone_photo_is_loaded_upright():
    """Phones store portrait shots sideways plus an orientation tag; ignoring it screens the art rotated."""
    import numpy as np
    from PIL import Image as PILImage

    import halftoner as ht

    upright = PILImage.new("RGB", (40, 90))
    for y in range(90):  # a vertical ramp, so a 90-degree error is unmistakable
        for x in range(40):
            upright.putpixel((x, y), (y * 2, y * 2, y * 2))

    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    upright.save(d / "plain.jpg", quality=95)

    sideways = upright.transpose(PILImage.Transpose.ROTATE_90)  # what the sensor writes
    exif = PILImage.Exif()
    exif[274] = 6  # "rotate 90 clockwise to display"
    sideways.save(d / "rotated.jpg", exif=exif, quality=95)
    assert PILImage.open(d / "rotated.jpg").size == (90, 40)  # stored sideways

    plain, rotated = ht.Image(d / "plain.jpg"), ht.Image(d / "rotated.jpg")
    assert rotated._lin.shape == plain._lin.shape == (90, 40)
    assert np.abs(rotated._lin - plain._lin).max() < 0.02  # same picture, not a transpose of it


def _wide_gamut_icc() -> bytes:
    """Pillow's sRGB profile with two primaries pushed out.

    Built by editing the numbers in a profile littleCMS already accepts, rather than shipping a
    binary fixture or hand-rolling a header: the structure stays valid, only the primaries move.
    """
    from PIL import ImageCms

    blob = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    for i in range(int.from_bytes(blob[128:132], "big")):  # walk the tag table
        off = 132 + i * 12
        if bytes(blob[off : off + 4]) in (b"rXYZ", b"bXYZ"):
            data = int.from_bytes(blob[off + 4 : off + 8], "big")  # XYZType: sig, reserved, 3 x s15Fixed16
            x = int.from_bytes(blob[data + 8 : data + 12], "big", signed=True)
            blob[data + 8 : data + 12] = int(x * 1.4).to_bytes(4, "big", signed=True)
    return bytes(blob)


def test_an_embedded_colour_profile_is_converted_to_srgb():
    """Phones tag Display P3, which shares sRGB's curve but not its primaries.

    Read raw, every saturated colour lands shifted -- which matters most where it is worst, on
    the strong flat colours an ink set gets chosen from.
    """
    import pathlib
    import tempfile

    import numpy as np
    from PIL import Image as PILImage

    import halftoner as ht

    d = pathlib.Path(tempfile.mkdtemp())
    patch = (222, 36, 40)
    PILImage.new("RGB", (32, 32), patch).save(d / "untagged.png")
    PILImage.new("RGB", (32, 32), patch).save(d / "tagged.png", icc_profile=_wide_gamut_icc())

    plain = float(ht.Image(d / "untagged.png", channel="r")._lin.mean())
    tagged = float(ht.Image(d / "tagged.png", channel="r")._lin.mean())
    assert plain == pytest.approx(float(srgb_to_linear(patch[0] / 255)), abs=1e-3)  # untagged: unchanged
    assert tagged > plain + 0.05, "the embedded profile was ignored"


def test_an_unusable_colour_profile_falls_back_to_srgb():
    """A profile littleCMS cannot build a transform from must not take the render down with it."""
    import numpy as np
    from PIL import Image as PILImage, ImageCms

    from halftoner.color import to_srgb_image

    img = PILImage.new("RGB", (8, 8), (222, 36, 40))
    img.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    assert np.array_equal(np.asarray(to_srgb_image(img)), np.asarray(img))
