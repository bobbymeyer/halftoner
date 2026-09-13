"""Recipe: the saved object. Every render is a render of it.

Constraints are computed and reported, never enforced: the interesting
territory is just past the line, and you can't aim there unless something
tells you where the line is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .canvas import Canvas
from .color import luma_linear
from .ink import InkSet
from .press import Press
from .screen import Plate, PlatePart, Screen, cell_range, lpi_to_pitch_mm
from .sources import Layered, Masked, clip_of, clip_regions
from .substrate import Substrate
from .transfer import ToneRange, Transfer
from .underbase import Underbase, erode

GEOMETRY_WARN = 300_000
MIN_ANGLE_SEPARATION = 15.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    kind: str = "info"  # ruling | min_dot | sampling | angle | geometry | coverage


@dataclass
class InkReport:
    name: str
    ruling_lpi: float
    angle: float
    shape: str
    joins: list[float]
    sampling_ratio: float | None
    min_dot: float | None
    max_dot: float | None
    min_dot_mm: float | None
    mean_plate: float
    mean_printed: float
    dots: int


@dataclass
class Report:
    canvas: Canvas
    substrate: Substrate
    inks: list[InkReport]
    palette: list
    checks: list[Check] = field(default_factory=list)
    profile: str | None = None
    tones: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        c, s = self.canvas, self.substrate
        w, h = c.size_px
        out = [
            f"press      {self.profile or '(no profile)'}",
            f"canvas     {c.width_mm:g} x {c.height_mm:g} mm @ {c.dpi:g} dpi = {w} x {h} px",
            f"substrate  {s.name}  paper {s.paper}  ceiling {s.ruling_ceiling_lpi:g} lpi",
            f"           provenance: {s.provenance}",
            "",
            "ink          lpi  angle  shape             joins        ratio  dot range     min dia  plate  printed  dots",
        ]
        for i in self.inks:
            joins = ",".join(f"{j:.0%}" for j in i.joins) or "-"
            ratio = f"{i.sampling_ratio:.2f}" if i.sampling_ratio is not None else "-"
            rng = f"{i.min_dot:.0%}-{i.max_dot:.0%}" if i.min_dot is not None else "-"
            dia = f"{i.min_dot_mm:.3f}mm" if i.min_dot_mm is not None else "-"
            out.append(
                f"{i.name[:12]:12} {i.ruling_lpi:4g} {i.angle:6.1f}  {i.shape[:16]:16}  {joins:11}  {ratio:>5}  {rng:12}  {dia:>7}  "
                f"{i.mean_plate:5.1%}  {i.mean_printed:6.1%}  {i.dots}"
            )
        out += ["", "palette"]
        for names, hexc, overridden in self.palette:
            label = " + ".join(names) if names else "paper"
            out.append(f"  {hexc}  {label}{'  (chosen)' if overridden else ''}")
        if self.tones:
            out += ["", "tone range (luminance: bare base -> solid as printed)"] + [f"  {t}" for t in self.tones]
        out += ["", "constraints (reported, not enforced)"]
        for ch in self.checks:
            out.append(f"  [{'ok' if ch.ok else 'PAST'}] {ch.name}: {ch.detail}")
        return "\n".join(out)


@dataclass
class Recipe:
    canvas: Canvas
    inks: InkSet
    screen: Screen
    source: object | None = None
    substrate: Substrate = field(default_factory=lambda: Substrate("plain"))
    press: Press = field(default_factory=Press)
    transfer: Transfer = field(default_factory=Transfer)
    seed: int | None = None
    profile: str | None = None  # name of the press profile this was built from, for the record
    underbase: Underbase | None = None  # plate printed first, derived from where the colors print

    def to_dict(self, base=None) -> dict:
        from .serialize import recipe_to_dict

        return recipe_to_dict(self, base)

    @classmethod
    def from_dict(cls, d: dict, base=None) -> Recipe:
        from .serialize import recipe_from_dict

        return recipe_from_dict(d, base)

    def save(self, path) -> None:
        from .serialize import save_recipe

        save_recipe(self, path)

    @classmethod
    def load(cls, path) -> Recipe:
        from .serialize import load_recipe

        return load_recipe(path)

    def __post_init__(self):
        if self.seed is not None:
            self.press.seed = self.seed
        self._plates: list[Plate] | None = None

    def ruling_for(self, ink) -> float:
        return ink.ruling_lpi or self.screen.ruling_lpi

    def source_for(self, ink):
        src = ink.source if ink.source is not None else self.source
        if src is None:
            raise ValueError(f"ink {ink.name!r} has no source and the recipe has none")
        return src

    @property
    def print_inks(self) -> InkSet:
        """Inks in print order: the underbase first when there is one, then the colors."""
        if self.underbase is None:
            return self.inks
        overprint = {tuple(k): v for k, v in self.inks.overprint.items()}
        return InkSet(self.underbase.ink, *self.inks.inks, overprint=overprint)

    def is_underbase(self, ink) -> bool:
        return self.underbase is not None and ink is self.underbase.ink

    def tone_range_for(self, ink) -> ToneRange:
        """Luminance of the bare substrate, and of a solid of `ink` as printed: with its opacity, over any underbase."""
        inks = self.print_inks
        primaries = inks.primaries(self.substrate.paper)
        mask = 1 << [i.name for i in inks].index(ink.name)
        if self.underbase is not None and not self.is_underbase(ink):
            mask |= 1  # the underbase is printed first, bit 0
        return ToneRange(float(luma_linear(primaries[0])), float(luma_linear(primaries[mask])))

    def plates(self) -> list[Plate]:
        """Plates in print order, matching print_inks."""
        if self._plates is None:
            colors = [self._build_plate(ink) for ink in self.inks]
            self._plates = ([self._build_underbase(colors)] if self.underbase else []) + colors
        return self._plates

    def invalidate(self) -> None:
        self._plates = None

    def _grid(self, ink) -> Plate:
        """An empty plate: this ink's rotated cell grid over the canvas, the bleed, and press travel."""
        sc, cv, press = self.screen, self.canvas, self.press
        pitch = lpi_to_pitch_mm(self.ruling_for(ink))
        # Cells out past the bleed (film and PDF print it) plus however far the press can move them.
        margin = press.misregistration * (1 + press.drift) * 2 + press.slur + pitch + cv.bleed_mm
        i0, j0, ni, nj = cell_range(cv, pitch, ink.angle, sc.origin, sc.phase, margin)
        empty = np.zeros((nj, ni))
        return Plate(ink, ink.shape or sc.shape, pitch, ink.angle, sc.origin, sc.phase, i0, j0, empty, empty)

    def _build_plate(self, ink) -> Plate:
        cv, press = self.canvas, self.press
        plate = self._grid(ink)
        pitch = plate.pitch_mm
        X, Y = plate.centers()
        src = self.source_for(ink)
        layers = src.sources if isinstance(src, Layered) else [src]
        samples = [(s.sample(X, Y, pitch, cv), s.kind) for s in layers]

        def printed_of(area):
            return np.where(area > 0, Transfer.printed_area(area, self.substrate, press.gain_curve), 0.0)

        tone = self.tone_range_for(ink)

        def layer_areas(bias, sel=slice(None)):
            return [self.transfer.plate_area(v[sel], k, ink, self.substrate, bias, tone) for v, k in samples]

        bias = 1.0
        on = np.flatnonzero((X >= 0) & (X < cv.width_mm) & (Y >= 0) & (Y < cv.height_mm))
        if ink.coverage is not None and on.size:
            sel = np.unravel_index(on[:: max(1, on.size // 200_000)], X.shape)  # bisect on a stride sample
            lo, hi = -6.0, 6.0  # log bias; more bias, less ink
            for _ in range(36):
                mid = (lo + hi) / 2
                covered = printed_of(np.maximum.reduce(layer_areas(np.exp(mid), sel))).mean()
                lo, hi = (mid, hi) if covered > ink.coverage else (lo, mid)
            bias = float(np.exp((lo + hi) / 2))

        areas = layer_areas(bias)
        plate.area = np.maximum.reduce(areas)
        plate.printed = printed_of(plate.area)
        plate.bias = bias
        clips = [clip_of(s) for s in layers]
        if all(c is None for c in clips):  # nothing to cut: one part is enough
            plate.parts = [PlatePart(plate.area, plate.printed)]
        else:
            plate.parts = [PlatePart(a, printed_of(a), c, clip_regions(s)) for a, c, s in zip(areas, clips, layers)]
        return plate

    def _layer_coverage(self, plate, X, Y, cell_mm):
        """Per layer of a color plate at arbitrary points: (printed coverage ignoring region cuts,
        region presence including edge cells, the layer's exact clip)."""
        ink = plate.ink
        src = self.source_for(ink)
        out = []
        for layer in src.sources if isinstance(src, Layered) else [src]:
            inner = layer
            presence = np.ones(np.shape(X), dtype=bool)
            while isinstance(inner, Masked):
                presence &= inner.region.signed_distance(X, Y) >= -cell_mm * np.sqrt(0.5)
                inner = inner.source
            if isinstance(inner, Layered):
                raise ValueError("an underbase needs Layered sources at the top level, not inside a Masked")
            area = self.transfer.plate_area(inner.sample(X, Y, cell_mm, self.canvas), inner.kind, ink,
                                            self.substrate, plate.bias, self.tone_range_for(ink))
            coverage = np.where(area > 0, Transfer.printed_area(area, self.substrate), 0.0)
            out.append((coverage, presence, clip_of(layer)))
        return out

    def _build_underbase(self, colors: list[Plate]) -> Plate:
        ub = self.underbase
        plate = self._grid(ub.ink)
        X, Y = plate.centers()
        radius = ub.choke_mm / plate.pitch_mm
        demand = np.zeros(X.shape)
        clips = []
        for color in colors:
            weight = ub.weights.get(color.ink.name, 1.0)
            if weight <= 0:
                continue
            for coverage, presence, clip in self._layer_coverage(color, X, Y, plate.pitch_mm):
                # Erode tone before applying region presence: region edges get their choke at full resolution.
                demand = np.maximum(demand, weight * np.where(presence, erode(coverage, radius), 0.0))
                clips.append(clip)
        own = replace(self.substrate, gain=ub.gain_curve)  # the base spreads on its own terms
        transfer = replace(self.transfer, compensate_gain=ub.compensate_gain)
        plate.area = transfer.plate_area(demand, "area", ub.ink, own)
        plate.printed = np.where(plate.area > 0, Transfer.printed_area(plate.area, own, self.press.gain_curve), 0.0)
        clip = None
        if clips and all(c is not None for c in clips):
            choke = ub.choke_mm
            clip = lambda x, y, inset=0.0: np.logical_or.reduce([c(x, y, inset + choke) for c in clips])  # noqa: E731
        plate.parts = [PlatePart(plate.area, plate.printed, clip)]
        return plate

    # --- outputs --------------------------------------------------------------------

    def render_png(self, path, supersample: int = 4, artifacts: bool = True) -> None:
        from .render.raster import composite, save_png

        save_png(composite(self, supersample=supersample, artifacts=artifacts), path, self.canvas.dpi)

    def render(self, target: str, out_dir=".", stem: str = "job", force: bool = False,
               supersample: int = 4, film_dpi: float = 1200, wedge: bool = True,
               output_condition: str | None = None, pdf_mode: str = "vector", bitmap_dpi: int = 2400):
        """Render through a target's policy. Returns an Outcome; raises ConstraintRefused unless forced."""
        from pathlib import Path

        from .policy import apply_policy

        job, outcome = apply_policy(self, target, force=force)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if target in ("screen", "pod"):
            path = out / f"{stem}{'_pod' if target == 'pod' else ''}.png"
            job.render_png(path, supersample=supersample)
            outcome.paths = [path]
        elif target == "svg":
            path = out / f"{stem}.svg"
            job.render_svg(path)
            outcome.paths = [path]
        elif target == "film":
            from .render.film import write_films

            outcome.paths = write_films(job, out, stem, dpi=film_dpi, wedge=wedge)
        elif target == "pdf":
            path = out / f"{stem}.pdf"
            job.render_pdf(path, output_condition=output_condition, mode=pdf_mode, bitmap_dpi=bitmap_dpi)
            outcome.paths = [path]
        return outcome

    def render_pdf(self, path, output_condition: str | None = None, marks: bool = True, mode: str = "vector",
                   bitmap_dpi: int = 2400) -> dict[str, int]:
        """PDF/X-1a spot separations, pre-screened and overprinted. Bypasses target policy."""
        from .render.pdf import write_pdf

        return write_pdf(self, path, output_condition=output_condition, marks=marks, mode=mode, bitmap_dpi=bitmap_dpi)

    def render_svg(self, path, artifacts: bool = False, precision: int = 2) -> dict[str, int]:
        from .render.svg import write_svg

        return write_svg(self, path, artifacts=artifacts, precision=precision)

    def report(self) -> Report:
        cv, sub = self.canvas, self.substrate
        rows, checks, tones, total_dots = [], [], [], 0
        for plate in self.plates():
            ink = plate.ink
            X, Y = plate.centers()
            on = (X >= 0) & (X < cv.width_mm) & (Y >= 0) & (Y < cv.height_mm)
            a, p = plate.area[on], plate.printed[on]
            partial = a[(a > 0) & (a < 1)]
            ratio = None if self.is_underbase(ink) else self.source_for(ink).sampling_ratio(plate.pitch_mm, cv)
            min_dot = float(partial.min()) if partial.size else None
            min_mm = plate.pitch_mm * np.sqrt(4 * min_dot / np.pi) if min_dot is not None else None
            dots = int((a > 0).sum())
            total_dots += dots
            if not self.is_underbase(ink):
                tr = self.tone_range_for(ink)
                mode = self.transfer.tone_mode(tr)
                meaning = "image white = base" if mode == "paper" else "image white = lighter end, black = darker"
                tones.append(f"{ink.name[:12]:12} {tr.base:.3f} -> {tr.solid:.3f}  {mode} ({meaning})")
            rows.append(
                InkReport(
                    ink.name, self.ruling_for(ink), ink.angle, plate.shape.name, plate.shape.join_points(), ratio,
                    min_dot, float(partial.max()) if partial.size else None, min_mm,
                    float(a.mean()) if a.size else 0.0, float(p.mean()) if p.size else 0.0, dots,
                )
            )
            lpi = self.ruling_for(ink)
            checks.append(Check(f"{ink.name} ruling <= substrate ceiling", lpi <= sub.ruling_ceiling_lpi,
                                f"{lpi:g} vs {sub.ruling_ceiling_lpi:g} lpi", "ruling"))
            if min_mm is not None and sub.min_printable_mm:
                checks.append(Check(f"{ink.name} smallest dot printable", min_mm >= sub.min_printable_mm,
                                    f"{min_mm:.3f} vs {sub.min_printable_mm:.3f} mm", "min_dot"))
            if ratio is not None:
                checks.append(Check(f"{ink.name} sampling ratio >= 1.0", ratio >= 1.0,
                                    f"{ratio:.2f} source px per cell", "sampling"))
            if ink.coverage is not None:
                got = rows[-1].mean_printed
                checks.append(Check(f"{ink.name} coverage target", abs(got - ink.coverage) <= 0.005,
                                    f"target {ink.coverage:.1%}, got {got:.1%} (bias {plate.bias:.2f})", "coverage"))

        inks = self.print_inks
        for a, b, sep in inks.angle_separations():
            ia = next(i for i in inks if i.name == a)
            ib = next(i for i in inks if i.name == b)
            if self.ruling_for(ia) == self.ruling_for(ib):
                checks.append(Check(f"{a}/{b} angle separation", sep >= MIN_ANGLE_SEPARATION,
                                    f"{sep:.1f} deg (moire below {MIN_ANGLE_SEPARATION:g})", "angle"))
        checks.append(Check("geometry count", total_dots <= GEOMETRY_WARN,
                            f"{total_dots:,} dots (browsers struggle past {GEOMETRY_WARN:,}; "
                            "for big jobs use the screen target or pdf bitmap mode)", "geometry"))
        return Report(cv, sub, rows, inks.palette(sub.paper), checks, self.profile, tones)
