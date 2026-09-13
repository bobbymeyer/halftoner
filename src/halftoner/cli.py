"""halftoner command line.

  halftoner profiles
  halftoner new job.json --profile newsprint_nominal --image photo.jpg --size 180x240 --ink red=#D6422B --ink blue=#1F3A63
  halftoner report job.json [--target film]
  halftoner render job.json --target screen|svg|pod|film|pdf [--out out] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canvas import Canvas
from .policy import TARGETS, ConstraintRefused, apply_policy
from .profile import PressProfile
from .recipe import Recipe
from .sources import Constant, Image


def _pair(text: str, sep: str = "="):
    if sep not in text:
        raise argparse.ArgumentTypeError(f"expected NAME{sep}VALUE, got {text!r}")
    return tuple(text.split(sep, 1))


def cmd_profiles(args) -> int:
    for name in PressProfile.available():
        p = PressProfile.load(name)
        print(f"{name:28} {p.ruling_lpi:g} lpi on {p.substrate.paper}  {p.provenance}")
    return 0


def cmd_new(args) -> int:
    profile = PressProfile.load(args.profile)
    w, h = (float(v) for v in args.size.lower().split("x"))
    canvas = Canvas.of(w, h, unit=args.unit, dpi=args.dpi, bleed=args.bleed)
    source = Image(Path(args.image).resolve()) if args.image else Constant(args.tint)
    overprint = {tuple(k.split("+")): v for k, v in (args.overprint or [])}
    job = profile.recipe(
        canvas=canvas,
        inks=profile.inks(*args.ink, overprint=overprint),
        source=source,
        seed=args.seed,
        screen={"ruling_lpi": args.ruling} if args.ruling else None,
        underbase=args.underbase,
    )
    job.save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_report(args) -> int:
    job = Recipe.load(args.recipe)
    print(job.report())
    try:
        _, outcome = apply_policy(job, args.target, force=False)
        print()
        print(outcome)
    except ConstraintRefused as e:
        print(f"\n{e}")
    return 0


def cmd_render(args) -> int:
    job = Recipe.load(args.recipe)
    try:
        outcome = job.render(
            args.target, args.out, stem=Path(args.recipe).stem, force=args.force,
            supersample=args.supersample, film_dpi=args.film_dpi, wedge=not args.no_wedge,
            output_condition=args.output_condition, pdf_mode=args.pdf_mode, bitmap_dpi=args.bitmap_dpi,
        )
    except ConstraintRefused as e:
        print(e, file=sys.stderr)
        return 2
    print(outcome)
    return 0


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def _patches(args) -> list:
    patches = []
    for spec in args.patch or []:
        box, _, nominal = spec.partition("=")
        patches.append((tuple(_floats(box)), float(nominal) if nominal else None))
    return patches


def cmd_calibrate(args) -> int:
    from .measure import measure_scan

    profile = PressProfile.load(args.profile)
    result = measure_scan(
        args.scan, n_inks=args.inks, dpi=args.dpi, item=args.item or "",
        names=args.names.split(",") if args.names else None,
        patches=_patches(args), wedge=tuple(_floats(args.wedge)) if args.wedge else None, patch_ink=args.patch_ink,
    )
    print(result.report())
    calibrated = result.calibrate_profile(profile, args.name)
    out = Path(args.out or f"{calibrated.name}.json")
    calibrated.save(out)
    before, after = profile.substrate, calibrated.substrate
    print(f"\ncalibrated {profile.name} -> {out}")
    print(f"  printed at a 50% plate dot: {float(before.gain(0.5)):.1%} -> {float(after.gain(0.5)):.1%}")
    print(f"  min dot {before.min_dot:.0%} -> {after.min_dot:.0%}, max dot {before.max_dot:.0%} -> {after.max_dot:.0%}")
    print("  provenance: DRAFT until checked")
    return 0


def cmd_measure(args) -> int:
    from .measure import measure_scan

    patches = _patches(args)
    result = measure_scan(
        args.scan, n_inks=args.inks, dpi=args.dpi, item=args.item or "",
        names=args.names.split(",") if args.names else None,
        colors=args.colors.split(",") if args.colors else None,
        key=args.key,
        paper_box_mm=tuple(_floats(args.paper_box)) if args.paper_box else None,
        patches=patches, wedge=tuple(_floats(args.wedge)) if args.wedge else None,
        patch_ink=args.patch_ink, min_lpi=args.min_lpi, max_lpi=args.max_lpi,
    )
    print(result.report())
    if args.profile_out:
        result.save_profile(args.profile_out, args.name)
        print(f"\nwrote draft profile {args.profile_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="halftoner", description="Ink-first halftones from press profiles.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("profiles", help="list press profiles").set_defaults(fn=cmd_profiles)

    n = sub.add_parser("new", help="write a recipe from a press profile")
    n.add_argument("out")
    n.add_argument("--profile", required=True)
    n.add_argument("--image", help="photo source; omit for a flat tint")
    n.add_argument("--tint", type=float, default=0.3, help="flat tint area when no image")
    n.add_argument("--size", required=True, help="WxH, e.g. 180x240")
    n.add_argument("--unit", default="mm", choices=["mm", "cm", "in"])
    n.add_argument("--dpi", type=float, default=300)
    n.add_argument("--bleed", type=float, default=0)
    n.add_argument("--ink", type=_pair, action="append", required=True, metavar="NAME=#HEX")
    n.add_argument("--overprint", type=_pair, action="append", metavar="A+B=#HEX")
    n.add_argument("--ruling", type=float, help="override the profile's ruling (lpi)")
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--underbase", action="store_true",
                   help="print a choked underbase first, from the profile's underbase settings (dark garments)")
    n.set_defaults(fn=cmd_new)

    r = sub.add_parser("report", help="print the report and what a target would do")
    r.add_argument("recipe")
    r.add_argument("--target", default="screen", choices=TARGETS)
    r.set_defaults(fn=cmd_report)

    d = sub.add_parser("render", help="render a recipe to a target")
    d.add_argument("recipe")
    d.add_argument("--target", default="screen", choices=TARGETS)
    d.add_argument("--out", default="out")
    d.add_argument("--force", action="store_true", help="render past a refusing policy")
    d.add_argument("--supersample", type=int, default=4)
    d.add_argument("--film-dpi", type=float, default=1200)
    d.add_argument("--no-wedge", action="store_true")
    d.add_argument("--output-condition", help="PDF/X registered characterization (default: substrate's, else FOGRA39)")
    d.add_argument("--pdf-mode", choices=["vector", "bitmap"], default="vector",
                   help="vector dots, or 1-bit image masks per separation (for large or fine-ruled jobs)")
    d.add_argument("--bitmap-dpi", type=int, default=2400, help="image mask resolution for --pdf-mode bitmap")
    d.set_defaults(fn=cmd_render)

    m = sub.add_parser("measure", help="measure a scan into a report and a draft press profile")
    m.add_argument("scan")
    m.add_argument("--inks", type=int, required=True, help="number of inks on the sheet (1-4)")
    m.add_argument("--dpi", type=float, help="scan resolution, if the file doesn't record it")
    m.add_argument("--names", help="comma-separated ink names, darkest first")
    m.add_argument("--colors", help="comma-separated ink colors (on white), if clustering can't find them")
    m.add_argument("--key", help="ink the others register against (default: darkest)")
    m.add_argument("--paper-box", metavar="X,Y,W,H", help="unprinted area in mm")
    m.add_argument("--patch", action="append", metavar="X,Y,W,H=NOMINAL", help="flat tint patch in mm")
    m.add_argument("--wedge", metavar="X,Y,PATCH_W", help="halftoner step wedge top-left and patch width, mm")
    m.add_argument("--patch-ink", help="ink the patches are printed in (solid color if no 100%% patch)")
    m.add_argument("--item", help="what was scanned: title, publisher, year")
    m.add_argument("--min-lpi", type=float, default=15)
    m.add_argument("--max-lpi", type=float, default=400)
    m.add_argument("--profile-out", help="write a draft press profile here")
    m.add_argument("--name", help="profile name (default: file stem)")
    m.set_defaults(fn=cmd_measure)

    c = sub.add_parser("calibrate", help="measure a printed step wedge into a copy of a press profile")
    c.add_argument("profile", help="profile name or path to start from")
    c.add_argument("scan")
    c.add_argument("--inks", type=int, default=1, help="inks on the sheet (1-4)")
    c.add_argument("--dpi", type=float, help="scan resolution, if the file doesn't record it")
    c.add_argument("--names", help="comma-separated ink names, darkest first")
    c.add_argument("--wedge", metavar="X,Y,PATCH_W", help="halftoner step wedge top-left and patch width, mm")
    c.add_argument("--patch", action="append", metavar="X,Y,W,H=NOMINAL", help="flat tint patch in mm")
    c.add_argument("--patch-ink", help="ink the wedge is printed in (solid color if no 100%% patch)")
    c.add_argument("--item", help="what was scanned")
    c.add_argument("--name", help="calibrated profile name (default: <profile>_calibrated)")
    c.add_argument("--out", help="where to write it (default: <name>.json)")
    c.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
