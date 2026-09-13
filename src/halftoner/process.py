"""CMYK process separations: a photo split into black, cyan, magenta and yellow plates.

This is a device separation, not a color-managed one. From the image's sRGB
values it takes the complementary cyan, magenta and yellow, replaces part of
their shared gray with black (gray component replacement, starting above a
threshold so highlights stay clean), normalizes what's left (undercolor
removal), and holds the total ink under a limit. For a separation that matches
a printing condition, separate with an ICC workflow and feed the channels in as
area sources; this is the honest default for making plates from a photo.

Process inks print at the classic angles (K 45, C 15, M 75, Y 0) in KCMY order,
and PDF/X writes them as DeviceCMYK rather than spot colors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import linear_to_srgb
from .ink import Ink, InkSet
from .sources import Image, Source, scale_source

# sRGB approximations of ISO coated process solids: for previews, not proofs.
PROCESS_COLORS = {"c": ("cyan", "#009FE3"), "m": ("magenta", "#E6007E"), "y": ("yellow", "#FFED00"),
                  "k": ("black", "#1D1D1B")}
PROCESS_ANGLES = {"k": 45.0, "c": 15.0, "m": 75.0, "y": 0.0}
PRINT_ORDER = ("k", "c", "m", "y")


@dataclass(frozen=True)
class CmykSeparation:
    gcr: float = 0.7  # how much of the gray component black takes over (0 = none, 1 = all)
    black_start: float = 0.2  # black begins once the gray component passes this
    tac: float = 3.0  # total area coverage limit: 3.0 = 300%

    def separate(self, rgb_linear: np.ndarray) -> np.ndarray:
        """(..., 4) nominal c, m, y, k areas for linear RGB (..., 3)."""
        cmy = 1.0 - linear_to_srgb(rgb_linear)
        gray = cmy.min(axis=-1)
        ramp = np.clip((gray - self.black_start) / max(1e-9, 1.0 - self.black_start), 0.0, 1.0)
        k = self.gcr * ramp * gray
        cmy = np.where(k[..., None] < 1.0 - 1e-9, (cmy - k[..., None]) / np.maximum(1.0 - k[..., None], 1e-9), 0.0)
        cmy = np.clip(cmy, 0.0, 1.0)
        total = cmy.sum(axis=-1)
        room = np.clip((self.tac - k) / np.maximum(total, 1e-9), 0.0, 1.0)  # black keeps its share; CMY give way
        cmy = np.where((total + k > self.tac)[..., None], cmy * room[..., None], cmy)
        return np.concatenate([cmy, k[..., None]], axis=-1)


class ProcessImage:
    """A photo loaded once and sampled as linear RGB; its channels share placement and prefiltering."""

    def __init__(self, path, fit: str = "cover", box=None, _channels=None):
        self.path, self.fit, self.box = path, fit, box
        self.channels = _channels or [Image(path, fit, ch, box) for ch in "rgb"]

    def sample_rgb(self, x, y, cell_mm, canvas) -> np.ndarray:
        return np.stack([c.sample(x, y, cell_mm, canvas) for c in self.channels], axis=-1)

    def sampling_ratio(self, cell_mm, canvas):
        return self.channels[0].sampling_ratio(cell_mm, canvas)

    def scaled(self, s: float) -> ProcessImage:
        box = tuple(v * s for v in self.box) if self.box else None
        return ProcessImage(self.path, self.fit, box, [scale_source(c, s) for c in self.channels])


@dataclass
class ProcessChannel(Source):
    image: ProcessImage
    channel: str  # "c" | "m" | "y" | "k"
    separation: CmykSeparation = field(default_factory=CmykSeparation)
    kind: str = "area"

    def __post_init__(self):
        if self.channel not in "cmyk" or len(self.channel) != 1:
            raise ValueError("channel must be one of c, m, y, k")

    def sample(self, x_mm, y_mm, cell_mm, canvas):
        rgb = self.image.sample_rgb(x_mm, y_mm, cell_mm, canvas)
        return self.separation.separate(rgb)[..., "cmyk".index(self.channel)]

    def sampling_ratio(self, cell_mm, canvas):
        return self.image.sampling_ratio(cell_mm, canvas)

    def scaled(self, s: float) -> ProcessChannel:
        return ProcessChannel(self.image.scaled(s), self.channel, self.separation)


def process_inks(path, separation: CmykSeparation | None = None, fit: str = "cover", box=None, profile=None,
                 order=PRINT_ORDER, overprint: dict | None = None) -> InkSet:
    """Black, cyan, magenta and yellow plates from one photo, at the classic angles.

    With a press profile, inks take its defaults (density, opacity) but keep the process angles.
    """
    separation = separation or CmykSeparation()
    image = ProcessImage(path, fit, box)
    inks = []
    for ch in order:
        name, color = PROCESS_COLORS[ch]
        source = ProcessChannel(image, ch, separation)
        if profile is not None:
            inks.append(profile.make_ink(name, color, 0, angle=PROCESS_ANGLES[ch], process=ch, source=source))
        else:
            inks.append(Ink(name, color, angle=PROCESS_ANGLES[ch], process=ch, source=source))
    return InkSet(*inks, overprint=overprint)
