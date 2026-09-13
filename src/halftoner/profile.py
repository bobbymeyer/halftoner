"""Press profiles: the unit of work.

Not lpi=65, gain=0.3, misreg=0.4 but press="uncoated_offset_nominal". A profile
bundles the substrate (paper, ceiling, gain, tone limits), the screen (ruling,
dot shape, angle set), the press (misregistration, drift, slur, uncorrected
gain), the tone model, and ink defaults, plus where the numbers came from.
Everything it supplies can be overridden per recipe, visibly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .cellfill import CellFill
from .curves import Curve
from .ink import Ink, InkSet
from .press import Press
from .screen import Screen
from .substrate import PROFILE_DIR, Substrate
from .transfer import Transfer


@dataclass
class PressProfile:
    name: str
    substrate: Substrate
    ruling_lpi: float
    shape: dict = field(default_factory=lambda: {"shape": "round"})
    angles: list[float] = field(default_factory=lambda: [45.0, 75.0, 15.0, 0.0])
    press: dict = field(default_factory=dict)  # Press fields except seed/key
    transfer: dict = field(default_factory=dict)  # Transfer fields
    ink: dict = field(default_factory=dict)  # Ink defaults: density, opacity
    provenance: str = "unspecified"
    notes: str = ""
    measurements: list[dict] = field(default_factory=list)  # scans the numbers came from

    @property
    def measured(self) -> bool:
        return self.provenance.upper().startswith("MEASURED")

    # --- io -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> PressProfile:
        d = dict(d)
        d["substrate"] = Substrate.from_dict(
            {"name": d["name"], "provenance": d.get("provenance", "unspecified"), **d["substrate"]}
        )
        screen = d.pop("screen", {})
        d.setdefault("ruling_lpi", screen.get("ruling_lpi"))
        d.setdefault("shape", screen.get("shape", {"shape": "round"}))
        if isinstance(d["shape"], str):
            d["shape"] = {"shape": d["shape"]}
        if "angles" in screen:
            d.setdefault("angles", screen["angles"])
        return cls(**d)

    def to_dict(self) -> dict:
        sub = self.substrate.to_dict()
        for k in ("name", "provenance"):
            sub.pop(k)
        return {
            "name": self.name,
            "provenance": self.provenance,
            "notes": self.notes,
            "measurements": self.measurements,
            "substrate": sub,
            "screen": {"ruling_lpi": self.ruling_lpi, "shape": self.shape, "angles": self.angles},
            "press": self.press,
            "transfer": self.transfer,
            "ink": self.ink,
        }

    @classmethod
    def load(cls, name_or_path: str | Path) -> PressProfile:
        p = Path(name_or_path)
        if not p.suffix:
            p = PROFILE_DIR / f"{name_or_path}.json"
        return cls.from_dict(json.loads(p.read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))

    # --- builders -------------------------------------------------------------------

    def make_screen(self, **overrides) -> Screen:
        return Screen(
            ruling_lpi=overrides.pop("ruling_lpi", self.ruling_lpi),
            shape=CellFill.from_spec(overrides.pop("shape", self.shape)),
            **overrides,
        )

    def make_press(self, seed: int = 0, **overrides) -> Press:
        kw = {**self.press, **overrides}
        if isinstance(kw.get("extra_gain"), (list, dict)):
            kw["extra_gain"] = Curve.from_spec(kw["extra_gain"])
        return Press(seed=seed, **kw)

    def make_transfer(self, **overrides) -> Transfer:
        return Transfer(**{**self.transfer, **overrides})

    def make_ink(self, name: str, color: str, index: int = 0, **overrides) -> Ink:
        """An ink with this press's defaults; angle taken from the profile's angle set by print order."""
        kw = {"angle": self.angles[index % len(self.angles)], **self.ink, **overrides}
        if "curve" in kw and not isinstance(kw["curve"], Curve):
            kw["curve"] = Curve.from_spec(kw["curve"])
        return Ink(name, color, **kw)

    def inks(self, *specs, overprint: dict | None = None) -> InkSet:
        """specs: Ink objects (kept as given) or (name, color) / (name, color, {overrides})."""
        made = []
        for i, s in enumerate(specs):
            if isinstance(s, Ink):
                made.append(s)
            else:
                name, color, *rest = s
                made.append(self.make_ink(name, color, i, **(rest[0] if rest else {})))
        return InkSet(*made, overprint=overprint)

    def recipe(self, canvas, inks, source=None, seed: int = 0, screen: dict | None = None,
               press: dict | None = None, transfer: dict | None = None, substrate: dict | None = None):
        from .recipe import Recipe

        sub = replace(self.substrate, **substrate) if substrate else self.substrate
        return Recipe(
            canvas=canvas,
            inks=inks if isinstance(inks, InkSet) else self.inks(*inks),
            screen=self.make_screen(**(screen or {})),
            source=source,
            substrate=sub,
            press=self.make_press(seed, **(press or {})),
            transfer=self.make_transfer(**(transfer or {})),
            seed=seed,
            profile=self.name,
        )
