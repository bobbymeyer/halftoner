"""Scan loading and array helpers. Scans are large, so conversion to linear light
happens on crops or in strips, never on the whole sheet at once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from ..canvas import MM_PER_INCH
from ..color import srgb_to_linear

_LUT = srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)


def load_scan(path: str | Path) -> tuple[np.ndarray, float | None]:
    """(H, W, 3) uint8 and the dpi recorded in the file, if any."""
    PILImage.MAX_IMAGE_PIXELS = None  # scans are legitimately huge
    img = PILImage.open(path)
    dpi = img.info.get("dpi")
    rgb = np.asarray(img.convert("RGB"))
    if not dpi or dpi[0] <= 1:
        return rgb, None
    d = float(dpi[0])
    # PNG stores whole pixels per metre, so 150 dpi comes back as 150.012.
    return rgb, float(round(d)) if abs(d - round(d)) <= 0.001 * d + 0.01 else d


def to_linear(u8: np.ndarray) -> np.ndarray:
    return _LUT[u8]


def crop_linear(u8: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    return to_linear(u8[y0 : y0 + h, x0 : x0 + w])


def downsample_linear(u8: np.ndarray, factor: int, strip: int = 64) -> np.ndarray:
    """Block-average in linear light by an integer factor."""
    factor = max(1, int(factor))
    H, W = (u8.shape[0] // factor) * factor, (u8.shape[1] // factor) * factor
    out = np.empty((H // factor, W // factor, 3), dtype=np.float32)
    rows = strip * factor
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        block = to_linear(u8[r0:r1, :W])
        out[r0 // factor : r1 // factor] = block.reshape(
            (r1 - r0) // factor, factor, W // factor, factor, 3
        ).mean(axis=(1, 3))
    return out


def gaussian_blur(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur of a 2-D array via the frequency domain (periodic edges)."""
    if sigma_px <= 0:
        return img
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    kernel = np.exp(-2 * np.pi**2 * sigma_px**2 * (fx**2 + fy**2))
    return np.fft.irfft2(np.fft.rfft2(img) * kernel, s=img.shape).astype(np.float32)


def mm_box_to_px(box_mm, dpi: float) -> tuple[int, int, int, int]:
    """(x, y, w, h) in mm -> (y0, x0, h, w) in pixels."""
    k = dpi / MM_PER_INCH
    x, y, w, h = box_mm
    return round(y * k), round(x * k), max(1, round(h * k)), max(1, round(w * k))
