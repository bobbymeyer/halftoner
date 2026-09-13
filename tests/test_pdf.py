"""PDF/X-1a separations: structure, spot separations, overprint, placement."""

import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image as PILImage
from pypdf import PdfReader

import halftoner as ht
from halftoner.policy import ConstraintRefused
from halftoner.render.pdf import PT_PER_MM, SLUG_MM, write_pdf


def _job(bleed=3.0, ceiling=150):
    red = ht.Ink("Warm Red", "#D6422B", angle=15, source=ht.Masked(ht.Constant(1.0), ht.Rect(0, 0, 20, 10)))
    blue = ht.Ink("prussian", "#1F3A63", angle=75, source=ht.Gradient(0.1, 0.9))
    return ht.Recipe(
        canvas=ht.Canvas.of(40, 30, dpi=150, bleed=bleed),
        inks=ht.InkSet(red, blue, overprint={("Warm Red", "prussian"): "#3A2036"}),
        screen=ht.Screen(ruling_lpi=40),
        substrate=ht.Substrate("t", ruling_ceiling_lpi=ceiling, output_condition="FOGRA29"),
    )


@pytest.fixture
def pdf(tmp_path):
    job = _job()
    path = tmp_path / "job.pdf"
    return job, path, write_pdf(job, path)


def test_pdfx_1a_structure(pdf):
    _, path, _ = pdf
    assert path.read_bytes().startswith(b"%PDF-1.3")
    reader = PdfReader(path)
    info = reader.metadata
    assert info["/GTS_PDFXVersion"] == "PDF/X-1:2001"
    assert info["/GTS_PDFXConformance"] == "PDF/X-1a:2001"
    assert info["/Trapped"] == "/False"
    assert info["/Title"] and info["/CreationDate"] and info["/ModDate"]
    assert "#3A2036" in info["/Keywords"]  # the chosen overprint is noted, not silently lost
    assert len(reader.trailer["/ID"]) == 2

    intent = reader.trailer["/Root"]["/OutputIntents"][0].get_object()
    assert intent["/S"] == "/GTS_PDFX"
    assert intent["/OutputConditionIdentifier"] == "FOGRA29"
    assert intent["/RegistryName"] == "http://www.color.org"

    page = reader.pages[0]
    k = PT_PER_MM
    trim, bleed, media = ([float(v) for v in box] for box in (page.trimbox, page.bleedbox, page.mediabox))
    assert trim[2] - trim[0] == pytest.approx(40 * k, abs=0.01)
    assert trim[3] - trim[1] == pytest.approx(30 * k, abs=0.01)
    assert bleed[0] == pytest.approx(trim[0] - 3 * k, abs=0.01)
    assert bleed[3] == pytest.approx(trim[3] + 3 * k, abs=0.01)
    assert media[0] <= bleed[0] and media[1] <= bleed[1] and media[2] >= bleed[2] and media[3] >= bleed[3]


def test_one_overprinting_spot_separation_per_ink(pdf):
    _, path, counts = pdf
    page = PdfReader(path).pages[0]
    spaces = page["/Resources"]["/ColorSpace"]
    seps = [spaces[key] for key in spaces]
    assert all(cs[0] == "/Separation" and cs[2] == "/DeviceCMYK" for cs in seps)
    names = {str(cs[1]) for cs in seps}
    assert "/prussian" in names and "/All" in names and any("Warm" in n for n in names)

    gs = page["/Resources"]["/ExtGState"]["/GS0"]
    assert bool(gs["/OP"]) and bool(gs["/op"]) and gs["/OPM"] == 1

    content = page.get_contents().get_data().decode("ascii")
    assert "/GS0 gs" in content
    assert not re.search(r"\b(rg|RG|k|K|sc|SC|g|G)\b", content)  # no device RGB, CMYK or gray: spot only

    sections = re.split(r"/CS\d+ cs 1 scn", content)
    assert [s.count(" m ") for s in sections[1:]] == [counts["Warm Red"], counts["prussian"]]
    assert all(n > 0 for n in counts.values())
    assert "re W n" in sections[1]  # the masked red plate is cut by its region


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="poppler not installed")
def test_renders_upright_and_in_place(tmp_path):
    job = ht.Recipe(
        canvas=ht.Canvas.of(40, 30, dpi=150),
        inks=ht.InkSet(ht.Ink("k", "#000000", angle=45, source=ht.Masked(ht.Constant(1.0), ht.Rect(0, 0, 20, 10)))),
        screen=ht.Screen(ruling_lpi=40),
        substrate=ht.Substrate("t", output_condition="FOGRA39"),
    )
    path = tmp_path / "k.pdf"
    write_pdf(job, path, marks=False)
    subprocess.run(["pdftoppm", "-r", "254", "-gray", "-png", "-singlefile", str(path), str(tmp_path / "k")], check=True)
    img = np.asarray(PILImage.open(tmp_path / "k.png").convert("L"))
    m = round(SLUG_MM * 10)  # 10 px/mm; no bleed
    assert img.shape[0] == pytest.approx((30 + 2 * SLUG_MM) * 10, abs=1)  # poppler rounds page size up
    assert img.shape[1] == pytest.approx((40 + 2 * SLUG_MM) * 10, abs=1)
    assert img[m + 5 : m + 95, m + 5 : m + 195].mean() < 30  # solid in the top-left 20 x 10 mm
    assert img[m + 150 : m + 290, m + 250 : m + 390].mean() > 250  # bare paper elsewhere


def test_pdf_target_refuses_like_film(tmp_path):
    job = _job(bleed=0.0, ceiling=30)
    with pytest.raises(ConstraintRefused):
        job.render("pdf", tmp_path)
    out = job.render("pdf", tmp_path, force=True)
    assert out.paths[0].suffix == ".pdf" and out.paths[0].exists()


def test_unregistered_output_condition_is_refused(tmp_path):
    with pytest.raises(ValueError, match="registered"):
        write_pdf(_job(), tmp_path / "x.pdf", output_condition="MyPress")
