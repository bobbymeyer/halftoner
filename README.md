# halftoner

Ink-first halftones from press profiles. A screen is a fill function over a grid; a job is an ink set; a recipe is the artifact and every output is a render of it.

```sh
uv sync
uv run pytest

uv run halftoner profiles
uv run halftoner new job.json --profile newsprint_nominal --image photo.jpg --size 180x240 --dpi 150 \
    --ink warm_red=#D6422B --ink prussian=#1F3A63 --overprint warm_red+prussian=#3A2036 --seed 7
uv run halftoner report job.json --target film      # report + what the film target would do
uv run halftoner render job.json --target screen    # screen | svg | pod | film  [--force]

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
  "press": {"misregistration": 0.3, "drift": 0.5, "slur": 0.04, "press_angle": 90, "extra_gain": 0.0},
  "transfer": {"yule_nielsen_n": 1.8, "compensate_gain": true},
  "ink": {"density": 0.9, "opacity": 0.0}
}
```

`gain` is a number (TVI at 50%) or measured `[[nominal, printed], ...]` pairs. Inks take angles from the profile's angle set in print order. Anything can be overridden per recipe: `profile.recipe(..., press={"misregistration": 0.5})`.

The bundled profiles are labeled `NOMINAL`. A profile named for a tradition should come from measured scans (see `_template_measured.json.example`): ruling counted against trim size, angles read off a rotated crop, overprints sampled, plate walk measured. The film target's step wedge closes the gain loop: print it, measure the patches, feed the pairs back as `gain`.

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

| Check | screen | svg | pod | film |
|---|---|---|---|---|
| Ruling ≤ substrate ceiling | report | report | cap | refuse |
| Smallest dot printable | report | report | report | refuse |
| Sampling ratio ≥ 1.0 | report | report | report | refuse |
| Ink angle separation ≥ 15° | report | report | report | refuse |
| Geometry count | – | warn | – | – |
| Coverage target reached | report | report | report | report |

`--force` / `render(force=True)` renders past a refusal and marks it in the outcome.

## Status against the spec

| # | Feature | State |
|---|---|---|
| 1–9 | Tone chain, ruling/angle/origin, dot shapes with joins, tone limits, gain curves, constant/gradient/region sources, single path per ink, sampling-ratio report, seeds | done |
| 10–11 | Ink set, overprint overrides, composite in ink space | done |
| 12 | Press artifacts | misregistration + drift + slur; density variance and trap gaps not yet |
| 13 | Anti-curves, tone compression | done |
| 14 | Substrate / press profiles | done, nominal only until measured |
| 15 | Constraint engine | done: computed, reported, policy-gated per target |
| 16 | Supersampled raster export | done |
| 17 | Step wedge emitter + curve ingest | wedge on film sheets; ingest via `gain` pairs |
| 18 | Film positive export | done (registration marks, labels, mirrored, 1-bit) |
| 22 | Mean coverage as a settable target | done (`Ink(coverage=0.2)`) |
| 19–21, 23–24 | Underbase/choke, garment base, grid-commensurate ruling, FM screening, PDF/X | not yet |
| — | resvg rasterizing of SVG output, density variance, trap gaps | not yet |
