# halftoner

Ink-first halftones from press profiles. A screen is a fill function over a grid; a job is an ink set; a recipe is the artifact and every output is a render of it.

```sh
uv sync
uv run pytest

uv run halftoner profiles
uv run halftoner new job.json --profile newsprint_nominal --image photo.jpg --size 180x240 --dpi 150 \
    --ink warm_red=#D6422B --ink prussian=#1F3A63 --overprint warm_red+prussian=#3A2036 --seed 7
uv run halftoner report job.json --target film      # report + what the film target would do
uv run halftoner render job.json --target screen    # screen | svg | pod | film | pdf  [--force]

uv run python examples/two_ink_poster.py [photo.jpg]  # the Python surface

uv run halftoner measure scan.tif --inks 2 --dpi 1200 --item "title, publisher, year" \
    --wedge 12,200,6 --patch-ink ink2 --profile-out draft.json   # scan -> report + DRAFT profile
```

## Pipeline

```
PressProfile ── substrate · screen (ruling, shape, angle set) · press · tone model · ink defaults · provenance
  │  builds
  ▼
Recipe (saved as JSON: canvas, inks, overprints, sources, all params, seed)
  │
Source (image | constant | gradient | function, optionally Masked / Layered)
  │  sampled at each cell center, box-prefiltered to cell size
  ▼
Transfer   tone → area (Yule–Nielsen vs. the ink's solid) → coverage bias → ink curve → compression
           → inverse substrate gain → highlight drop / shadow snap           = plate area
  ▼
Plate      one per ink: rotated grid, per-cell area; printed = gain(plate) + press gain
  ▼
Report     sampling ratio, dot range, min dot diameter, coverage, palette, checks (always computed)
  ▼
Policy     per target: report | warn | cap | refuse (forceable)
  ▼
Targets    screen  supersampled, press artifacts, ink bitmask → Neugebauer primaries → linear box average → sRGB
           pod     screen, with ruling capped at the substrate ceiling
           svg     one <path> per ink in mm; arcs below the join, traced polygons above
           film    per ink, 1-bit, no AA, no artifacts, mirrored; crop marks, reg targets, label, step wedge
           pdf     PDF/X-1a:2001: one overprinting spot separation per ink, pre-screened vector dots,
                   trim/bleed boxes, registration-color marks, registered output intent
```

## Press profiles

`src/halftoner/profiles/*.json`. One file holds everything a press implies:

```json
{
  "name": "newsprint_nominal",
  "provenance": "NOMINAL - generic newsprint ranges, not measured from scans",
  "measurements": [],
  "substrate": {"paper": "#E6DFCC", "ruling_ceiling_lpi": 85, "gain": 0.28, "min_dot": 0.08, "max_dot": 0.85,
                "min_printable_mm": 0.1, "compression": [0.1, 0.85]},
  "screen": {"ruling_lpi": 65, "shape": "round", "angles": [45, 75, 15, 0]},
  "press": {"misregistration": 0.3, "drift": 0.5, "slur": 0.04, "press_angle": 90,
            "density_variance": 0.06, "trap_gap": 0.0, "extra_gain": 0.0},
  "transfer": {"yule_nielsen_n": 1.8, "compensate_gain": true},
  "ink": {"density": 0.9, "opacity": 0.0}
}
```

`gain` is a number (TVI at 50%) or measured `[[nominal, printed], ...]` pairs. Inks take angles from the profile's angle set in print order. Anything can be overridden per recipe: `profile.recipe(..., press={"misregistration": 0.5})`.

The bundled profiles are labeled `NOMINAL`. A profile named for a tradition should come from measured scans (see `_template_measured.json.example`): ruling counted against trim size, angles read off a rotated crop, overprints sampled, plate walk measured. The film target's step wedge closes the gain loop: print it, measure the patches, feed the pairs back as `gain`.

## Press artifacts

All seeded, all per ink (never per RGB channel), applied by the screen and pod targets and left off film.

| Artifact | Parameter | Model |
|---|---|---|
| Misregistration | `misregistration`, `drift`, `key` | per-plate constant offset plus low-frequency walk; the key plate stays put. Region cuts move with their plate |
| Slur | `slur`, `press_angle` | each dot smeared along the press direction |
| Density variance | `density_variance` | ink film thickness wanders across the sheet (±fraction, low frequency); each present ink's density scales Beer–Lambert style on top of its primary, so chosen overprints keep their character |
| Trap gap | `trap_gap` | masked regions are choked by half the gap, opening a paper hairline where regions meet. With misregistration, butted regions already gap on one side and overlap on the other |
| Uncorrected gain | `extra_gain` | an anti-curve applied on press |

Masked regions are cut at full resolution, the way a tint was cut from film: edge cells carry the region's value and their dots are sliced at the true edge (raster, film, and SVG clip paths).

## Measuring scans

`halftoner measure` turns a scan of real print into a report and a draft profile. Scan at 10+ pixels per screen cell: 1200 dpi covers screens up to 120 lpi; use 2400 dpi for finer work. Below that, blurred dot edges swamp the pure-ink pixels the separation relies on, and the report will say so.

| Reading | How |
|---|---|
| Paper | median of the lightest 2% of pixels, or `--paper-box` |
| Inks and overprints | pixels clustered in optical-density space into 2ⁿ primaries, each walked to its density peak; single inks are the subset whose sums best explain the rest. Overprints that aren't subtractive are flagged (chosen color, opaque ink, trap) |
| Ruling and angle | Fourier peak of each ink's unmixed amount map, pooled over a 3×3 grid of windows |
| Misregistration | each plate descreened and phase-correlated against the key plate in tiles: median = offset, spread = drift. Needs shared structure between plates |
| Gain and tone limits | effective Murray–Davies coverage on `--patch` boxes or a halftoner `--wedge`, measured on the solid's densest channel |

The draft's provenance is `DRAFT`; change it to `MEASURED` only after checking the numbers against the scan. Not measured: dot shape (set it from a loupe crop), slur, density variance.

Validated against renders with known parameters (`tests/test_measure.py`): ruling within 1%, angle within 0.5°, ink colors, registration offsets within 0.05 mm, gain within 2 points.

## Constraint policy

| Check | screen | svg | pod | film | pdf |
|---|---|---|---|---|---|
| Ruling ≤ substrate ceiling | report | report | cap | refuse | refuse |
| Smallest dot printable | report | report | report | refuse | refuse |
| Sampling ratio ≥ 1.0 | report | report | report | refuse | refuse |
| Ink angle separation ≥ 15° | report | report | report | refuse | refuse |
| Geometry count | – | warn | – | – | – |
| Coverage target reached | report | report | report | report | report |

## PDF/X

`--target pdf` writes PDF/X-1a:2001 (ISO 15930-1) separations for a printer:

- One `Separation` color space per ink, named for the ink; every dot is a 100% tint, so the RIP images the screen as drawn and never re-screens it. Each separation's DeviceCMYK alternate is a naive conversion for on-screen proofing only.
- Overprint on (`OP`, `op`, `OPM 1`) for everything: plates never knock each other out, and the press makes the overprint colors. A chosen overprint color is a property of the real inks, so it's recorded in the document's Keywords rather than encoded.
- TrimBox = canvas, BleedBox = canvas + bleed, 12 mm slug with crop marks and registration targets in the `All` color. Dots are clipped at the bleed.
- OutputIntent names a registered characterization without embedding a profile: `FOGRA39` (coated), `FOGRA29` (uncoated), `IFRA26` (newsprint). Set it per substrate (`output_condition`) or with `--output-condition`.
- No press artifacts, no fonts, no transparency; PDF 1.3.

Built to the standard and checked structurally and by rendering (`tests/test_pdf.py`), but not yet run through a PDF/X preflight (Acrobat Preflight, callas pdfToolbox). Preflight before sending a job.

`--force` / `render(force=True)` renders past a refusal and marks it in the outcome.

## Status against the spec

| # | Feature | State |
|---|---|---|
| 1–9 | Tone chain, ruling/angle/origin, dot shapes with joins, tone limits, gain curves, constant/gradient/region sources, single path per ink, sampling-ratio report, seeds | done |
| 10–11 | Ink set, overprint overrides, composite in ink space | done |
| 12 | Press artifacts | done: misregistration + drift, slur, density variance, trap gaps |
| 13 | Anti-curves, tone compression | done |
| 14 | Substrate / press profiles | done, nominal only until measured |
| 15 | Constraint engine | done: computed, reported, policy-gated per target |
| 16 | Supersampled raster export | done |
| 17 | Step wedge emitter + curve ingest | wedge on film sheets; ingest via `gain` pairs |
| 18 | Film positive export | done (registration marks, labels, mirrored, 1-bit) |
| 22 | Mean coverage as a settable target | done (`Ink(coverage=0.2)`) |
| 24 | PDF/X export with separations and overprint flags | done (PDF/X-1a:2001; not yet preflighted) |
| 19–21, 23 | Underbase/choke, garment base, grid-commensurate ruling, FM screening | not yet |
| — | resvg rasterizing of SVG output | not yet |
