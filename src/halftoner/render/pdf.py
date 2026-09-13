"""PDF/X-1a:2001 separations: one spot color per ink, pre-screened, overprinted.

The screen is already baked in, so every mark is a 100% tint of its ink's
Separation color space: the RIP images it solid and never re-screens it. Two
ways to carry the screen:

  vector  each dot is a path (circles as Beziers, traced polygons otherwise).
          Exact at any output resolution; size grows with dot count.
  bitmap  each separation is a 1-bit image mask at platesetter resolution over
          the bleed box, Flate-compressed and streamed to disk strip by strip.
          The usual way pre-screened art ships; size stays sane at fine rulings.

Overprint is on for every object, so plates never knock each other out and the
press makes the overprint colors, as it did. A chosen overprint color can't be
carried: it's a property of the real inks, so the file notes it instead.

Structure follows ISO 15930-1:2001 (PDF/X-1a:2001): PDF 1.3; spot and CMYK
color only; no transparency and no fonts; MediaBox, BleedBox and TrimBox; an
OutputIntent naming a registered characterization; Title, dates, Trapped and
the GTS_PDFX keys in Info; and a file ID. Each spot color's DeviceCMYK
alternate is a naive conversion, for on-screen proofing only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import zlib
from pathlib import Path

import numpy as np

from ..canvas import MM_PER_INCH
from ..color import cmyk_from_linear
from ..sources import Rect
from .bitmap import hit_rows
from .geometry import dot_geometry

PT_PER_MM = 72 / MM_PER_INCH
SLUG_MM = 12.0  # outside the bleed: crop marks and registration targets
KAPPA = 0.5522847498  # cubic Bezier quarter-circle handle length
RESERVED_SEPARATIONS = {"All", "None"}
MODES = ("vector", "bitmap")

# Registered characterizations (color.org) that PDF/X-1a allows without an embedded profile.
REGISTERED_CONDITIONS = {
    "FOGRA39": "Coated FOGRA39 (ISO 12647-2:2004)",
    "FOGRA29": "Uncoated FOGRA29 (ISO 12647-2:2004)",
    "IFRA26": "ISOnewspaper26v4 (ISO 12647-3:2004)",
}


def _name(s: str) -> str:
    """PDF name object, with delimiters and non-regular bytes #-escaped."""
    out = []
    for byte in s.encode("utf-8"):
        c = chr(byte)
        out.append(c if 33 <= byte <= 126 and c not in "#()<>[]{}/%" else f"#{byte:02X}")
    return "/" + "".join(out)


def _text(s: str) -> str:
    """PDF literal string (Latin-1, escaped)."""
    s = s.encode("latin-1", "replace").decode("latin-1")
    return "(" + s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


def _separation(name: str, cmyk) -> str:
    c1 = " ".join(f"{v:.4f}" for v in cmyk)
    return f"[/Separation {_name(name)} /DeviceCMYK << /FunctionType 2 /Domain [0 1] /C0 [0 0 0 0] /C1 [{c1}] /N 1 >>]"


def _circle_ops(circles: np.ndarray, precision: int) -> list[str]:
    if not len(circles):
        return []
    x, y, r = circles.T
    kr = KAPPA * r
    cols = [x + r, y,
            x + r, y + kr, x + kr, y + r, x, y + r,
            x - kr, y + r, x - r, y + kr, x - r, y,
            x - r, y - kr, x - kr, y - r, x, y - r,
            x + kr, y - r, x + r, y - kr, x + r, y]
    f = f"%.{precision}f"
    template = f"{f} {f} m " + " ".join([" ".join([f] * 6) + " c"] * 4) + " h"
    return [template % tuple(row) for row in np.column_stack(cols)]


def _polygon_ops(polygons: np.ndarray, keep: np.ndarray, precision: int) -> list[str]:
    f = f"%.{precision}f"
    point = f"{f} {f}"
    out = []
    for poly, k in zip(polygons, keep):
        pts = poly[k]
        out.append(f"{point % tuple(pts[0])} m " + " ".join(f"{point % tuple(p)} l" for p in pts[1:]) + " h")
    return out


def _region_ops(region) -> str:
    if isinstance(region, Rect):
        return f"{region.x:.4f} {region.y:.4f} {region.w:.4f} {region.h:.4f} re"
    pts = np.asarray(region.points, dtype=np.float64)
    signed = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if signed < 0:
        pts = pts[::-1]  # wind like `re`, so overlapping regions union under the nonzero rule
    return f"{pts[0, 0]:.4f} {pts[0, 1]:.4f} m " + " ".join(f"{x:.4f} {y:.4f} l" for x, y in pts[1:]) + " h"


def _marks(canvas, gap: float, length: float = 6.0, radius: float = 2.5) -> list[str]:
    """Crop marks at the trim corners and registration targets centred on each side."""
    w, h = canvas.width_mm, canvas.height_mm
    ops = ["0.1 w"]
    for cx in (0.0, w):
        for cy in (0.0, h):
            sx = -1 if cx == 0 else 1
            sy = -1 if cy == 0 else 1
            ops.append(f"{cx + sx * gap:.3f} {cy:.3f} m {cx + sx * (gap + length):.3f} {cy:.3f} l S")
            ops.append(f"{cx:.3f} {cy + sy * gap:.3f} m {cx:.3f} {cy + sy * (gap + length):.3f} l S")
    d = gap + length / 2
    for tx, ty in ((w / 2, -d), (w / 2, h + d), (-d, h / 2), (w + d, h / 2)):
        ops.append(_circle_ops(np.array([[tx, ty, radius]]), 3)[0] + " S")
        arm = radius + 1.5
        ops.append(f"{tx - arm:.3f} {ty:.3f} m {tx + arm:.3f} {ty:.3f} l S")
        ops.append(f"{tx:.3f} {ty - arm:.3f} m {tx:.3f} {ty + arm:.3f} l S")
    return ops


def write_pdf(recipe, path, output_condition: str | None = None, marks: bool = True, precision: int = 3,
              tolerance_mm: float = 0.005, mode: str = "vector", bitmap_dpi: int = 2400) -> dict[str, int]:
    """Write PDF/X-1a separations.

    Returns dots written per ink (vector) or inked pixels per ink (bitmap).
    `tolerance_mm` bounds how far a traced vector dot edge strays from the true
    one; 5 microns is well under what a 2400 dpi platesetter resolves.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    cv = recipe.canvas
    condition = output_condition or recipe.substrate.output_condition or "FOGRA39"
    if condition not in REGISTERED_CONDITIONS:
        raise ValueError(f"output condition {condition!r} isn't a registered characterization this writer knows "
                         f"({', '.join(REGISTERED_CONDITIONS)}); an unregistered one would need an embedded ICC profile")
    plates = recipe.plates()
    for p in plates:
        if p.ink.name in RESERVED_SEPARATIONS:
            raise ValueError(f"ink name {p.ink.name!r} is reserved in PDF separations")

    k = PT_PER_MM
    b, w, h = cv.bleed_mm, cv.width_mm, cv.height_mm
    off = b + SLUG_MM
    media_w, media_h = w + 2 * off, h + 2 * off
    bitmap = mode == "bitmap"
    counts: dict[str, int] = {}

    # --- page content ---------------------------------------------------------------
    ops = ["/GS0 gs"]
    if bitmap:
        # Whole pixels at bitmap_dpi over the bleed box, top-left aligned to it.
        ppm = bitmap_dpi / MM_PER_INCH
        width_px, height_px = round((w + 2 * b) * ppm), round((h + 2 * b) * ppm)
        width_pt, height_pt = width_px / bitmap_dpi * 72, height_px / bitmap_dpi * 72
        x_pt, y_pt = SLUG_MM * k, (media_h - SLUG_MM) * k - height_pt
        for i in range(len(plates)):
            ops.append(f"q /CS{i} cs 1 scn {width_pt:.4f} 0 0 {height_pt:.4f} {x_pt:.4f} {y_pt:.4f} cm /Im{i} Do Q")
    # Vector content in canvas millimetres, y down, origin at the trim's top-left corner.
    ops += ["q", f"{k:.6f} 0 0 {-k:.6f} {off * k:.4f} {(media_h - off) * k:.4f} cm"]
    if marks:
        ops += ["q", "/CSAll CS 1 SCN", *_marks(cv, gap=b + 2.0), "Q"]
    if not bitmap:
        ops.append(f"q {-b:.4f} {-b:.4f} {w + 2 * b:.4f} {h + 2 * b:.4f} re W n")  # nothing past the bleed
        for i, plate in enumerate(plates):
            ops.append(f"/CS{i} cs 1 scn")
            counts[plate.ink.name] = 0
            for part in plate.parts:
                g = dot_geometry(plate, cv, part, use_printed=False, margin_mm=b, tolerance_mm=tolerance_mm)
                if not g.count:
                    continue
                counts[plate.ink.name] += g.count
                dots = _circle_ops(g.circles, precision) + _polygon_ops(g.polygons, g.keep, precision)
                if part.regions:
                    ops += ["q " + " ".join(_region_ops(r) for r in part.regions) + " W n", *dots, "f Q"]
                else:
                    ops += [*dots, "f"]
        ops.append("Q")
    ops.append("Q")
    content = zlib.compress("\n".join(ops).encode("ascii"), 6)

    # --- document objects -------------------------------------------------------------
    now = _dt.datetime.now(_dt.timezone.utc).strftime("D:%Y%m%d%H%M%S+00'00'")
    screens = "; ".join(f"{p.ink.name}: {recipe.ruling_for(p.ink):g} lpi at {p.angle_deg:g} deg, {p.shape.name}"
                        for p in plates)
    carried = f"1-bit image masks at {bitmap_dpi} dpi" if bitmap else "vector dots"
    note = f"Pre-screened spot separations ({carried}), overprinted."
    if recipe.inks.overprint:
        chosen = ", ".join("+".join(sorted(combo)) + f" {color}" for combo, color in recipe.inks.overprint.items())
        note += f" Chosen overprints ({chosen}) are properties of the real inks and are not encoded; proof on press."
    colorspaces = " ".join(f"/CS{i} {_separation(p.ink.name, cmyk_from_linear(p.ink.transmittance))}"
                           for i, p in enumerate(plates))
    box = lambda x0, y0, x1, y1: f"[{x0 * k:.3f} {y0 * k:.3f} {x1 * k:.3f} {y1 * k:.3f}]"  # noqa: E731
    images = {i: (8 + 2 * i, 9 + 2 * i) for i in range(len(plates))} if bitmap else {}
    xobjects = ("/XObject << " + " ".join(f"/Im{i} {num} 0 R" for i, (num, _) in images.items()) + " >> ") if images else ""

    header_objects = {
        1: "<< /Type /Catalog /Pages 2 0 R /OutputIntents [5 0 R] >>",
        2: "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (f"<< /Type /Page /Parent 2 0 R /MediaBox {box(0, 0, media_w, media_h)} "
            f"/BleedBox {box(SLUG_MM, SLUG_MM, SLUG_MM + w + 2 * b, SLUG_MM + h + 2 * b)} "
            f"/TrimBox {box(off, off, off + w, off + h)} "
            f"/Resources << /ColorSpace << {colorspaces} /CSAll {_separation('All', (1, 1, 1, 1))} >> {xobjects}"
            f"/ExtGState << /GS0 << /Type /ExtGState /OP true /op true /OPM 1 >> >> >> /Contents 4 0 R >>"),
        5: (f"<< /Type /OutputIntent /S /GTS_PDFX /OutputConditionIdentifier {_text(condition)} "
            f"/OutputCondition {_text(REGISTERED_CONDITIONS[condition])} /RegistryName (http://www.color.org) "
            f"/Info {_text(REGISTERED_CONDITIONS[condition])} >>"),
        6: (f"<< /Title {_text(Path(path).stem)} /Subject {_text(screens)} /Keywords {_text(note)} "
            f"/Creator (halftoner) /Producer (halftoner) /CreationDate ({now}) /ModDate ({now}) /Trapped /False "
            f"/GTS_PDFXVersion (PDF/X-1:2001) /GTS_PDFXConformance (PDF/X-1a:2001) >>"),
    }

    def mask_chunks(plate):
        """Flate-compressed 1-bit rows (1 = ink), rasterized and compressed strip by strip."""
        compressor = zlib.compressobj(6)
        inked_px = 0
        for _, _, _, inked in hit_rows(plate, -b, -b, width_px, height_px, bitmap_dpi):
            inked_px += int(inked.sum())
            data = compressor.compress(np.packbits(inked, axis=1).tobytes())
            if data:
                yield data
        counts[plate.ink.name] = inked_px
        yield compressor.flush()

    # --- write, streaming ---------------------------------------------------------------
    offsets: dict[int, int] = {}
    digest = hashlib.md5()
    with open(path, "wb") as fh:
        pos = 0

        def emit(data: bytes) -> None:
            nonlocal pos
            fh.write(data)
            digest.update(data)
            pos += len(data)

        def obj(num: int, body: str) -> None:
            offsets[num] = pos
            emit(f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1"))

        def stream(num: int, head: str, chunks, length_num: int) -> None:
            offsets[num] = pos
            emit(f"{num} 0 obj\n<< {head} /Length {length_num} 0 R >>\nstream\n".encode("latin-1"))
            n = 0
            for chunk in chunks:
                emit(chunk)
                n += len(chunk)
            emit(b"\nendstream\nendobj\n")
            obj(length_num, str(n))  # length known only after streaming

        emit(b"%PDF-1.3\n%\xe2\xe3\xcf\xd3\n")
        for num, body in header_objects.items():
            obj(num, body)
        stream(4, "/Filter /FlateDecode", [content], 7)
        for i, (num, length_num) in images.items():
            head = (f"/Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} /ImageMask true "
                    f"/BitsPerComponent 1 /Decode [1 0] /Filter /FlateDecode")
            stream(num, head, mask_chunks(plates[i]), length_num)

        size = max(offsets) + 1
        xref_at = pos
        emit(f"xref\n0 {size}\n0000000000 65535 f \n".encode("ascii"))
        emit(b"".join(f"{offsets[n]:010d} 00000 n \n".encode("ascii") for n in range(1, size)))
        file_id = digest.hexdigest()
        emit((f"trailer\n<< /Size {size} /Root 1 0 R /Info 6 0 R /ID [<{file_id}> <{file_id}>] >>\n"
              f"startxref\n{xref_at}\n%%EOF\n").encode("ascii"))
    return counts
