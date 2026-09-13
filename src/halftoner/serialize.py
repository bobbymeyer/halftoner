"""Recipe <-> JSON. The recipe is the artifact: store source + params, render on demand.

Saved recipes are self-contained (the full substrate is written out, with the
profile name kept for the record). Image paths are stored relative to the
recipe file. Function sources and custom shapes are code, so they refuse to
serialize rather than silently dropping out.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .canvas import Canvas
from .cellfill import CellFill
from .curves import Curve
from .ink import Ink, InkSet
from .press import Press
from .screen import Screen
from .sources import Constant, Gradient, Image, Layered, Masked, Polygon, Rect
from .substrate import Substrate
from .transfer import Transfer

FORMAT = 1


# --- sources ------------------------------------------------------------------------


def region_to_spec(r) -> dict:
    if isinstance(r, Rect):
        return {"rect": [r.x, r.y, r.w, r.h]}
    return {"polygon": [list(p) for p in r.points]}


def region_from_spec(d: dict):
    if "rect" in d:
        return Rect(*d["rect"])
    return Polygon(tuple(tuple(p) for p in d["polygon"]))


def source_to_spec(src, base: Path | None = None):
    if src is None:
        return None
    if isinstance(src, Constant):
        return {"type": "constant", "value": src.value, "kind": src.kind}
    if isinstance(src, Gradient):
        return {"type": "gradient", "start": src.start, "end": src.end, "p0": list(src.p0),
                "p1": list(src.p1) if src.p1 else None, "kind": src.kind}
    if isinstance(src, Image):
        if callable(src.channel):
            raise TypeError("Image with a callable channel can't be serialized")
        path = Path(src.path).resolve()
        if base is not None:
            try:
                path = path.relative_to(base.resolve())
            except ValueError:
                pass
        return {"type": "image", "path": str(path), "fit": src.fit, "channel": src.channel,
                "box": list(src.box) if src.box else None}
    if isinstance(src, Masked):
        return {"type": "masked", "source": source_to_spec(src.source, base), "region": region_to_spec(src.region)}
    if isinstance(src, Layered):
        return {"type": "layered", "sources": [source_to_spec(s, base) for s in src.sources]}
    raise TypeError(f"{type(src).__name__} sources are code and can't be serialized")


def source_from_spec(d, base: Path | None = None):
    if d is None:
        return None
    t = d["type"]
    if t == "constant":
        return Constant(d["value"], d.get("kind", "area"))
    if t == "gradient":
        return Gradient(d["start"], d["end"], tuple(d["p0"]), tuple(d["p1"]) if d.get("p1") else None,
                        d.get("kind", "area"))
    if t == "image":
        path = Path(d["path"])
        if not path.is_absolute() and base is not None:
            path = base / path
        return Image(path, d.get("fit", "cover"), d.get("channel", "luma"), tuple(d["box"]) if d.get("box") else None)
    if t == "masked":
        return Masked(source_from_spec(d["source"], base), region_from_spec(d["region"]))
    if t == "layered":
        return Layered([source_from_spec(s, base) for s in d["sources"]])
    raise ValueError(f"unknown source type {t!r}")


# --- job objects --------------------------------------------------------------------


def ink_to_dict(ink: Ink, base=None) -> dict:
    return {
        "name": ink.name, "color": ink.color, "density": ink.density, "angle": ink.angle,
        "curve": ink.curve.to_spec(), "opacity": ink.opacity,
        "source": source_to_spec(ink.source, base),
        "shape": ink.shape.to_spec() if ink.shape else None,
        "ruling_lpi": ink.ruling_lpi, "coverage": ink.coverage,
    }


def ink_from_dict(d: dict, base=None) -> Ink:
    d = dict(d)
    d["curve"] = Curve.from_spec(d.get("curve"))
    d["source"] = source_from_spec(d.get("source"), base)
    d["shape"] = CellFill.from_spec(d["shape"]) if d.get("shape") else None
    return Ink(**d)


def press_to_dict(p: Press) -> dict:
    d = asdict(p)
    d["extra_gain"] = p.extra_gain.to_spec() if isinstance(p.extra_gain, Curve) else p.extra_gain
    return d


def press_from_dict(d: dict) -> Press:
    d = dict(d)
    if isinstance(d.get("extra_gain"), (list, dict)):
        d["extra_gain"] = Curve.from_spec(d["extra_gain"])
    return Press(**d)


def recipe_to_dict(r, base: Path | None = None) -> dict:
    return {
        "halftoner": FORMAT,
        "profile": r.profile,
        "seed": r.press.seed,
        "canvas": asdict(r.canvas),
        "substrate": r.substrate.to_dict(),
        "screen": {"ruling_lpi": r.screen.ruling_lpi, "shape": r.screen.shape.to_spec(),
                   "origin": list(r.screen.origin), "phase": list(r.screen.phase)},
        "press": press_to_dict(r.press),
        "transfer": asdict(r.transfer),
        "inks": [ink_to_dict(i, base) for i in r.inks],
        "overprint": [{"inks": sorted(k), "color": c} for k, c in r.inks.overprint.items()],
        "source": source_to_spec(r.source, base),
    }


def recipe_from_dict(d: dict, base: Path | None = None):
    from .recipe import Recipe

    if d.get("halftoner") != FORMAT:
        raise ValueError(f"unsupported recipe format {d.get('halftoner')!r}")
    sc = d["screen"]
    return Recipe(
        canvas=Canvas(**d["canvas"]),
        inks=InkSet(*(ink_from_dict(i, base) for i in d["inks"]),
                    overprint={tuple(o["inks"]): o["color"] for o in d.get("overprint", [])}),
        screen=Screen(sc["ruling_lpi"], CellFill.from_spec(sc["shape"]), tuple(sc["origin"]), tuple(sc["phase"])),
        source=source_from_spec(d.get("source"), base),
        substrate=Substrate.from_dict(d["substrate"]),
        press=press_from_dict(d["press"]),
        transfer=Transfer(**d["transfer"]),
        seed=d.get("seed"),
        profile=d.get("profile"),
    )


def save_recipe(r, path) -> None:
    path = Path(path)
    path.write_text(json.dumps(recipe_to_dict(r, path.parent), indent=2) + "\n")


def load_recipe(path):
    path = Path(path)
    return recipe_from_dict(json.loads(path.read_text()), path.parent)
