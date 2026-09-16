# roadmap

## open

| Work | Issue |
| --- | --- |
| Hybrid AM/FM and blue-noise FM screening | [#3](https://github.com/bobbymeyer/halftoner/issues/3) |
| Preflight the PDF/X-1a output | [#4](https://github.com/bobbymeyer/halftoner/issues/4) |

## not planned

**resvg rasterizing.** Compositing happens in ink space, through the
Neugebauer primaries. resvg would rasterize the SVG's multiply-blend preview,
which is RGB layer compositing. The screen and pod targets already composite
separations with overprint overrides, opacity and press artifacts; large
vector jobs go through `--target pdf --pdf-mode bitmap`. SVG stays a preview
and an editable vector export.

## standing caveat

Every bundled profile is marked `NOMINAL`: plausible ranges for the stock,
not measurements of any press. `halftoner measure` and `halftoner calibrate`
replace them from a scan, and write `DRAFT until checked` into the
provenance field.
