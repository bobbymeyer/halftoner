"""The looks in the README's gallery, one panel each.

    uv run python examples/styles.py --portrait P --night N --colour C --graphic G

Each option is a photograph the gallery uses for the panels it suits: a
portrait, a dark night scene, a saturated colour photograph, and a piece of
flat graphic art. Any that is missing falls back to a synthetic stand-in, so
the script runs anywhere — except --colour, whose panel is a four-ink
separation and is simply skipped, because a stand-in would be demonstrating
the stand-in rather than the separation.

Every panel is the same square canvas, the same seed and the same 4x
supersample, so what differs between them is only what is named in the
caption. Writes to examples/gallery/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import halftoner as ht

OUT = Path(__file__).parent / "gallery"
SEED = 7
CANVAS = ht.Canvas.of(70, 70, unit="mm", dpi=203.2)  # 8 px/mm -> 560 x 560

UNCOATED = ht.PressProfile.load("uncoated_offset_nominal")
NEWS = ht.PressProfile.load("newsprint_nominal")
GARMENT = ht.PressProfile.load("plastisol_dark_garment_nominal")
CLEAN = {"misregistration": 0.0, "drift": 0.0, "slur": 0.0, "density_variance": 0.0}


def _stand_in(kind: str):
    """A synthetic field standing in for a photograph that wasn't supplied."""
    if kind == "graphic":  # a hard-edged mark on bare paper
        return ht.Function(lambda x, y: np.where(np.abs(x - 35) / 1.7 + np.abs(y - 35) / 3.4 < 12,
                                                 0.18, 0.97), kind="tone")
    dark = 0.55 if kind == "night" else 0.0

    def field(x, y):
        r = np.hypot((x - 32) / 21, (y - 30) / 26)
        v = 0.93 - 0.80 * np.exp(-(r**2) * 1.7) - dark * (y / 70)
        return np.clip(v + 0.06 * np.sin(x / 7) * np.cos(y / 9), 0.03, 0.99)

    return ht.Function(field, kind="tone")


def source(path: str | None, kind: str, box=None):
    """`box` is the canvas rect the photo is fitted to; one taller than the canvas and
    started above it shows the lower part of a portrait-shaped source instead of its middle."""
    return ht.Image(path, fit="cover", box=box) if path else _stand_in(kind)


def render(slug: str, job) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    job.render_png(OUT / f"{slug}.png", supersample=4)
    print(f"  {slug}.png")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, help_text in [("portrait", "a face, for the ruling panels"),
                            ("night", "a dark scene, for the duotone"),
                            ("colour", "a saturated photograph, for the CMYK separation"),
                            ("graphic", "flat art with a hard edge, for the line screen")]:
        ap.add_argument(f"--{name}", help=help_text)
    args = ap.parse_args(argv)
    portrait, night, colour, graphic = args.portrait, args.night, args.colour, args.graphic

    # 1. The canonical halftone: one ink on newsprint, at a ruling a newspaper could hold.
    render("newsprint", NEWS.recipe(
        canvas=CANVAS, inks=NEWS.inks(("k", "#141414")), source=source(portrait, "portrait"),
        seed=SEED, screen={"ruling_lpi": 65, "shape": {"shape": "elliptical", "ratio": 1.2}}))

    # 2. The same picture with the screen coarse enough to become the subject.
    render("coarse", UNCOATED.recipe(
        canvas=CANVAS, inks=UNCOATED.inks(("k", "#16181D")), source=source(portrait, "portrait"),
        seed=SEED, screen={"ruling_lpi": 16, "shape": "round"}, press=CLEAN))

    # 3. Two spot inks on their own curves, and an overprint colour that was chosen
    #    rather than computed -- the thing compositing in ink space is for.
    render("duotone", UNCOATED.recipe(
        canvas=CANVAS,
        inks=UNCOATED.inks(
            ("midnight", "#16243D", {"curve": ht.Curve.from_points([(0.35, 0.30), (0.75, 0.72)])}),
            ("rust", "#B4572B", {"curve": ht.Curve.from_points([(0.40, 0.09), (0.80, 0.26)])}),
            overprint={("midnight", "rust"): "#0F1622"}),
        source=source(night, "night"), seed=SEED,
        screen={"ruling_lpi": 50, "shape": {"shape": "elliptical", "ratio": 1.4}}, press=CLEAN))

    # 4. A device separation: KCMY at the classic angles, 70% grey replacement, 300% ink limit.
    if colour:
        inks = ht.process_inks(colour, ht.CmykSeparation(tac=UNCOATED.substrate.tac or 3.0),
                               profile=UNCOATED)
        render("process", UNCOATED.recipe(canvas=CANVAS, inks=inks, source=None, seed=SEED,
                                          screen={"ruling_lpi": 55}, press=CLEAN))
    else:
        print("  (no --colour photo; skipping the CMYK panel)")

    # 5. A line screen carries tone by line weight alone and never joins across.
    render("line", UNCOATED.recipe(
        canvas=CANVAS, inks=UNCOATED.inks(("ink", "#CC4038")), source=source(graphic, "graphic"),
        seed=SEED, screen={"ruling_lpi": 34, "shape": "line"}, press=CLEAN))

    # 6. On a near-black shirt both inks are lighter than the garment, so tone_range
    #    auto reads the picture as range: more ink where the image is lighter. The wall
    #    behind a portrait becomes the ink and the dark shirt becomes bare cotton.
    render("garment", GARMENT.recipe(
        canvas=CANVAS, inks=GARMENT.inks(("bone", "#EFE8DA"), ("amber", "#D08A2E")),
        source=source(portrait, "portrait"), seed=SEED, screen={"ruling_lpi": 40},
        press={"misregistration": 0.3, "drift": 0.3, "slur": 0.0, "density_variance": 0.05},
        underbase={"choke_mm": 1.0}))

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
