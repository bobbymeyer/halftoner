"""Shared test configuration.

`pdftoppm` (poppler) is what the PDF tests rasterize with, and it is a system
binary that `uv sync` does not install. The tests that need it skip when it is
missing -- which is how the only checks that look at what actually lands on a
PDF page went unrun for as long as nobody happened to have poppler. CI sets
HALFTONER_REQUIRE_POPPLER=1 so a missing binary is an error there, not a skip.
"""

import os
import shutil

import pytest

POPPLER = "pdftoppm"
HAVE_POPPLER = shutil.which(POPPLER) is not None

needs_poppler = pytest.mark.skipif(not HAVE_POPPLER, reason="poppler not installed")


def pytest_configure(config):
    if os.environ.get("HALFTONER_REQUIRE_POPPLER") == "1" and not HAVE_POPPLER:
        raise pytest.UsageError(
            f"HALFTONER_REQUIRE_POPPLER=1 but {POPPLER} is not on PATH, so the PDF render "
            "tests would skip instead of run. Install poppler-utils (apt), poppler (brew), "
            "or unset HALFTONER_REQUIRE_POPPLER to allow skipping."
        )
