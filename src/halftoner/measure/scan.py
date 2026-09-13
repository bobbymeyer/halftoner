"""Measure a scan into a report and a draft press profile.

Everything here is a reading, not a verdict: the draft is provenance DRAFT and
only becomes MEASURED when someone has checked it against the loupe.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..canvas import MM_PER_INCH
from ..color import linear_to_hex
from .common import load_scan, mm_box_to_px
from .register import RegistrationMeasurement, measure_registration
from .screen import ScreenMeasurement, measure_screens
from .separate import InkModel, estimate_paper, fit_inks
from .tone import PatchReading, gain_pairs, measure_patches, tone_limits, wedge_boxes

MIN_PX_PER_CELL = 10.0  # below this, blurred dot edges swamp the pure-ink pixels separation relies on
RESIDUAL_WARN = 0.15


@dataclass
class ScanMeasurement:
    path: str
    item: str
    dpi: float
    size_mm: tuple[float, float]
    model: InkModel
    screens: list[ScreenMeasurement | None]
    registration: dict[str, RegistrationMeasurement | None]
    key: str
    patches: list[PatchReading] = field(default_factory=list)
    patch_ink: str | None = None
    warnings: list[str] = field(default_factory=list)

    # --- text -----------------------------------------------------------------------

    def report(self) -> str:
        m = self.model
        out = [
            f"scan       {Path(self.path).name}  {self.dpi:g} dpi  {self.size_mm[0]:.1f} x {self.size_mm[1]:.1f} mm",
            f"paper      {linear_to_hex(m.paper)}",
            "",
            "inks",
        ]
        for i, name in enumerate(m.names):
            s = self.screens[i]
            screen = (f"{s.ruling_lpi:6.1f} lpi @ {s.angle:4.1f} deg  (+-{s.spread_lpi:.1f} lpi, +-{s.spread_angle:.1f} deg, "
                      f"{s.windows} windows, {s.px_per_cell:.1f} px/cell, conf {s.confidence:.0f})") if s else "no screen found"
            reg = self.registration.get(name)
            if name == self.key:
                where = "key plate"
            elif reg is None:
                where = "registration: no shared structure"
            else:
                where = (f"offset ({reg.dx_mm:+.3f}, {reg.dy_mm:+.3f}) mm = {reg.magnitude_mm:.3f} mm, "
                         + (f"walk {reg.drift_mm:.3f} +- {reg.drift_se_mm:.3f} mm"
                            f"{'' if reg.walk_resolved else ' (unresolved)'}"
                            if np.isfinite(reg.drift_se_mm) else "walk unmeasured")
                         + f" (tile noise {reg.noise_mm:.3f} mm, {len(reg.tiles)} tiles)")
            out.append(f"  {name:8} {m.ink_hex(i)}  {screen}")
            out.append(f"  {'':8} {'':7}  {where}")
        out += ["", "primaries (sampled)"]
        for mask in sorted(m.primaries, key=lambda k: (bin(k).count("1"), k)):
            if mask not in m.observed:
                extra = "  not observed: predicted subtractively"
            elif mask in m.residual:
                extra = f"  departs from subtractive by {m.residual[mask]:.2f} D"
            else:
                extra = ""
            out.append(f"  {linear_to_hex(m.primaries[mask])}  {m.label(mask):20} {m.coverage[mask]:5.1%} of pixels{extra}")
        if self.patches:
            out += ["", f"tone ({self.patch_ink or 'patches'}; effective Murray-Davies coverage)"]
            for r in self.patches:
                nom = f"{r.nominal:4.0%}" if r.nominal is not None else "   ?"
                out.append(f"  {nom} -> {r.coverage:6.1%}")
            lo, hi = tone_limits(self.patches)
            out.append(f"  highlight dots hold from {lo:.0%}" if lo is not None else "  no highlight dot held")
            out.append(f"  whites stay open up to {hi:.0%}" if hi is not None else "  no shadow white stayed open")
        if self.warnings:
            out += ["", "warnings"] + [f"  {w}" for w in self.warnings]
        return "\n".join(out)

    # --- draft profile ----------------------------------------------------------------

    def draft_profile(self, name: str) -> dict:
        m = self.model
        found = [s for s in self.screens if s]
        rulings = [s.ruling_lpi for s in found]
        offsets = [r.magnitude_mm for n, r in self.registration.items() if r and n != self.key]
        drifts = [r.drift_mm for n, r in self.registration.items() if r and n != self.key and r.walk_resolved]
        misreg = float(np.median(offsets)) if offsets else 0.0
        pairs = gain_pairs(self.patches) if self.patches else []
        lo, hi = tone_limits(self.patches) if self.patches else (None, None)

        measurement = {
            "item": self.item or "unspecified",
            "scan": Path(self.path).name,
            "scan_dpi": self.dpi,
            "size_mm": [round(v, 2) for v in self.size_mm],
            "date": _dt.date.today().isoformat(),
            "method": "halftoner measure",
            "paper": linear_to_hex(m.paper),
            "inks": [
                {
                    "name": n,
                    "color_on_white": m.ink_hex(i),
                    "ruling_lpi": round(self.screens[i].ruling_lpi, 2) if self.screens[i] else None,
                    "angle": round(self.screens[i].angle, 2) if self.screens[i] else None,
                    "offset_mm": ([round(self.registration[n].dx_mm, 3), round(self.registration[n].dy_mm, 3)]
                                  if self.registration.get(n) else None),
                    "drift_mm": round(self.registration[n].drift_mm, 3) if self.registration.get(n) else None,
                    "registration_noise_mm": round(self.registration[n].noise_mm, 3) if self.registration.get(n) else None,
                }
                for i, n in enumerate(m.names)
            ],
            "primaries": {m.label(k): linear_to_hex(v) for k, v in sorted(m.primaries.items())},
            "overprint_residual_D": {m.label(k): round(v, 3) for k, v in m.residual.items()},
            "gain_pairs": pairs,
            "patch_ink": self.patch_ink,
            "warnings": self.warnings,
        }
        notes = [
            "Draft from `halftoner measure`. Check each number against the scan before changing provenance to MEASURED.",
            "ruling_ceiling_lpi is the finest ruling observed; the stock may hold finer.",
            "shape is not measured: set it by eye from a loupe crop.",
            "gain is effective (Murray-Davies) coverage, so yule_nielsen_n is 1." if pairs else "gain not measured: no patches given.",
        ]
        return {
            "name": name,
            "provenance": "DRAFT - measured from scans, unverified",
            "notes": " ".join(notes),
            "measurements": [measurement],
            "substrate": {
                "paper": linear_to_hex(m.paper),
                "ruling_ceiling_lpi": round(max(rulings), 1) if rulings else 0,
                "gain": pairs or 0.0,
                "min_dot": lo if lo is not None else 0.0,
                "max_dot": hi if hi is not None else 1.0,
                "min_printable_mm": 0.0,
                "compression": None,
            },
            "screen": {
                "ruling_lpi": round(float(np.median(rulings)), 1) if rulings else 0,
                "shape": "round",
                "angles": [round(s.angle, 1) for s in found],
            },
            "press": {
                "misregistration": round(misreg, 3),
                "drift": round(min(1.0, max(drifts) / misreg), 2) if drifts and misreg > 0 else 0.35,
                "slur": 0.0,
                "press_angle": 90,
                "extra_gain": 0.0,
            },
            "transfer": {"yule_nielsen_n": 1.0, "compensate_gain": True},
            "ink": {"density": 1.0, "opacity": 0.0},
        }

    def save_profile(self, path, name: str | None = None) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.draft_profile(name or path.stem), indent=2) + "\n")


def measure_scan(path, n_inks: int, dpi: float | None = None, item: str = "", names: list[str] | None = None,
                 colors: list[str] | None = None, key: str | None = None, paper_box_mm=None,
                 patches=None, wedge: tuple[float, float, float] | None = None, patch_ink: str | None = None,
                 seed: int = 0, min_lpi: float = 15, max_lpi: float = 400) -> ScanMeasurement:
    u8, file_dpi = load_scan(path)
    dpi = dpi or file_dpi
    if not dpi:
        raise ValueError("scan has no dpi recorded; pass dpi")
    rng = np.random.default_rng(seed)
    H, W = u8.shape[:2]
    warnings: list[str] = []

    paper = estimate_paper(u8, mm_box_to_px(paper_box_mm, dpi) if paper_box_mm else None, rng)
    model = fit_inks(u8, n_inks, paper, rng, colors=colors, names=names)
    screens = measure_screens(u8, model, dpi, min_lpi=min_lpi, max_lpi=max_lpi)

    for name, s in zip(model.names, screens):
        if s is None:
            warnings.append(f"{name}: no screened area found (all solid, all empty, or not a halftone)")
            continue
        if s.px_per_cell < MIN_PX_PER_CELL:
            warnings.append(f"{name}: only {s.px_per_cell:.1f} px per cell; rescan at >= "
                            f"{MIN_PX_PER_CELL * s.ruling_lpi:.0f} dpi for trustworthy dots")
        if s.spread_angle > 2 or s.spread_lpi > 0.03 * s.ruling_lpi:
            warnings.append(f"{name}: windows disagree (+-{s.spread_lpi:.1f} lpi, +-{s.spread_angle:.1f} deg); "
                            "check for harmonics near 50% or mixed screens")
    for mask, r in model.residual.items():
        if r > RESIDUAL_WARN:
            warnings.append(f"{model.label(mask)}: overprint departs from subtractive by {r:.2f} D "
                            "(chosen color, opaque ink, or trapping?)")
    for mask in range(1 << n_inks):
        if mask not in model.observed:
            warnings.append(f"{model.label(mask)}: not observed on the sheet, predicted subtractively; if these inks "
                            "do overprint here, the separation may be wrong (rescan finer, or pass colors)")

    key_idx = model.names.index(key) if key else 0
    rulings = [s.ruling_lpi for s in screens if s]
    if n_inks > 1 and rulings:
        registration = measure_registration(u8, model, dpi, min(rulings), key=key_idx)
    else:
        registration = {model.names[key_idx]: RegistrationMeasurement(0.0, 0.0, 0.0)}
    for name, reg in registration.items():
        if reg is None or name == model.names[key_idx]:
            continue
        if len(reg.tiles) < 5:
            warnings.append(f"{name}: registration from only {len(reg.tiles)} tile(s); offset is rough and walk unmeasured")
        elif reg.noise_mm > 0.15:
            warnings.append(f"{name}: registration tiles scatter by {reg.noise_mm:.2f} mm; the plates share too few "
                            "edges (smooth tones mislead correlation), check the offset by eye")
        elif not reg.walk_resolved:
            warnings.append(f"{name}: walk of {reg.drift_mm:.2f} +- {reg.drift_se_mm:.2f} mm is within what tile "
                            "scatter produces; left out of the draft profile")

    boxes = list(patches or [])
    if wedge:
        boxes += wedge_boxes(*wedge)
    readings: list[PatchReading] = []
    if boxes:
        solid = model.solid_linear(model.names.index(patch_ink)) if patch_ink else None
        readings = measure_patches(u8, dpi, boxes, paper, solid)

    return ScanMeasurement(str(path), item, dpi, (W * MM_PER_INCH / dpi, H * MM_PER_INCH / dpi), model, screens,
                           registration, model.names[key_idx], readings, patch_ink, warnings)
