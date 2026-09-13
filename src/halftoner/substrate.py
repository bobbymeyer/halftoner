"""Substrate: what the paper (or mesh, or garment) can hold.

Paper stock, screen mesh and a DTG head all resolve to this: a ruling ceiling,
a gain curve, tone limits and a base color. Profiles carry provenance, because a
profile named for a tradition is only honest if it was measured from scans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .curves import Curve

PROFILE_DIR = Path(__file__).parent / "profiles"


@dataclass(frozen=True)
class Substrate:
    name: str
    paper: str = "#FFFFFF"
    ruling_ceiling_lpi: float = 150.0
    gain: Curve = field(default_factory=Curve.identity)
    min_dot: float = 0.0  # below this nominal area the dot drops out (highlight drop)
    max_dot: float = 1.0  # above this the white closes up (shadow snap)
    min_printable_mm: float = 0.0  # smallest dot diameter the stock holds
    compression: tuple[float, float] | None = None  # (lo, hi) tone range the stock can reproduce
    output_condition: str | None = None  # registered characterization for PDF/X, e.g. FOGRA29
    tac: float | None = None  # total area coverage limit for the stock: 3.0 = 300%
    provenance: str = "unspecified"
    notes: str = ""

    def suggest(self, viewing_distance: float) -> float:
        """Coarsest ruling whose dots sit below ~2 arcminutes at `viewing_distance` metres, capped."""
        inches = viewing_distance * 1000 / 25.4
        return min(self.ruling_ceiling_lpi, round(1719 / inches, 1))

    @classmethod
    def from_dict(cls, d: dict) -> Substrate:
        d = dict(d)
        d["gain"] = Curve.from_spec(d.get("gain"))
        if d.get("compression"):
            d["compression"] = tuple(d["compression"])
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "paper": self.paper, "ruling_ceiling_lpi": self.ruling_ceiling_lpi,
            "gain": self.gain.to_spec(), "min_dot": self.min_dot, "max_dot": self.max_dot,
            "min_printable_mm": self.min_printable_mm,
            "compression": list(self.compression) if self.compression else None,
            "output_condition": self.output_condition,
            "tac": self.tac,
            "provenance": self.provenance, "notes": self.notes,
        }

    @classmethod
    def load(cls, name_or_path: str | Path) -> Substrate:
        """A bare substrate file, or the substrate section of a press profile."""
        p = Path(name_or_path)
        if not p.suffix:
            p = PROFILE_DIR / f"{name_or_path}.json"
        d = json.loads(p.read_text())
        if "substrate" in d:
            d = {"name": d["name"], "provenance": d.get("provenance", "unspecified"), **d["substrate"]}
        return cls.from_dict(d)

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
