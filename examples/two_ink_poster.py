"""Two spot inks, a chosen overprint, a flat tint panel, and a press that walks.

    uv run python examples/two_ink_poster.py [photo.jpg]

Without a photo, a synthetic tone field stands in.
"""

import sys
from pathlib import Path

import numpy as np

import halftoner as ht

OUT = Path("out")
OUT.mkdir(exist_ok=True)

SEED = 7
CANVAS = ht.Canvas.of(180, 240, unit="mm", dpi=150)
SUBSTRATE = ht.Substrate.load("uncoated_offset_nominal")
RULING = 30  # coarse on purpose, past the viewing-distance suggestion, so the dots read
ANGLE_BASE = 22.5
MISREG = 0.35  # mm
PHOTO_BOX = ht.Rect(0, 0, 180, 170)
BAND = ht.Rect(0, 176, 180, 64)

if len(sys.argv) > 1:
    photo = ht.Image(sys.argv[1], fit="cover", box=(PHOTO_BOX.x, PHOTO_BOX.y, PHOTO_BOX.w, PHOTO_BOX.h))
else:
    def tone_field(x, y):
        r = np.hypot((x - 90) / 55, (y - 80) / 70)
        return np.clip(0.9 - 0.8 * np.exp(-(r**2) * 2.2) * (0.6 + 0.4 * np.sin(x / 9)), 0.02, 1)
    photo = ht.Function(tone_field, kind="tone")

# Duotone: same image, two inks, different curves. Red carries the mids, blue the shadows.
warm_red = ht.Ink(
    "warm_red", "#D6422B", density=0.85, angle=ANGLE_BASE,
    curve=ht.Curve.from_points([(0.3, 0.45), (0.7, 0.75)]),
    source=ht.Layered([
        ht.Masked(photo, PHOTO_BOX),
        ht.Masked(ht.Constant(0.25), BAND),  # flat 25% tint: no image behind it, same screen path
    ]),
)
prussian = ht.Ink(
    "prussian", "#1F3A63", density=0.9, angle=ANGLE_BASE + 30,
    curve=ht.Curve.from_points([(0.4, 0.15), (0.8, 0.7)]),
    source=ht.Masked(photo, PHOTO_BOX),
)

job = ht.Recipe(
    canvas=CANVAS,
    substrate=SUBSTRATE,
    inks=ht.InkSet(warm_red, prussian, overprint={("warm_red", "prussian"): "#3A2036"}),
    screen=ht.Screen(ruling_lpi=RULING, shape=ht.Elliptical(1.4)),
    press=ht.Press(misregistration=MISREG, slur=0.04, extra_gain=0.05),
    transfer=ht.Transfer(yule_nielsen_n=1.6),
    seed=SEED,
)

print(f"suggested ruling at 1.5 m: {SUBSTRATE.suggest(1.5)} lpi (using {RULING})\n")
print(job.report())
job.render_png(OUT / "two_ink_poster.png", supersample=4)
job.render_svg(OUT / "two_ink_poster.svg")
print(f"\nwrote {OUT / 'two_ink_poster.png'} and {OUT / 'two_ink_poster.svg'}")
