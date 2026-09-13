import pathlib

import numpy as np
import pytest
from PIL import Image as PILImage

import halftoner as ht
from halftoner.cli import main
from halftoner.policy import POLICIES, ConstraintRefused, apply_policy
from halftoner.recipe import CHECK_KINDS
from halftoner.render.film import MARGIN_MM


def _job(ruling=40, source=None, ceiling=150):
    return ht.Recipe(
        canvas=ht.Canvas.of(20, 10, dpi=254),  # 10 px/mm
        inks=ht.InkSet(ht.Ink("k", "#000", angle=45)),
        screen=ht.Screen(ruling_lpi=ruling),
        source=source or ht.Constant(0.4),
        substrate=ht.Substrate("t", ruling_ceiling_lpi=ceiling),
    )


def test_screen_reports_what_film_refuses(tmp_path):
    job = _job(ruling=120, ceiling=85)
    _, outcome = apply_policy(job, "screen")
    assert [a for _, a in outcome.actions] == ["report"]
    with pytest.raises(ConstraintRefused):
        job.render("film", tmp_path)
    forced = job.render("film", tmp_path, force=True, film_dpi=254)
    assert forced.forced and len(forced.paths) == 1


def test_pod_caps_ruling_at_substrate_ceiling(tmp_path):
    job = _job(ruling=120, ceiling=85)
    capped, outcome = apply_policy(job, "pod")
    assert outcome.capped == {"k": (120, 85)}
    assert capped.ruling_for(capped.inks.inks[0]) == 85
    assert job.ruling_for(job.inks.inks[0]) == 120  # original recipe untouched


def test_film_is_one_bit_mirrored_with_margin(tmp_path):
    left_half = ht.Masked(ht.Constant(1.0), ht.Rect(0, 0, 10, 10))
    out = _job(source=left_half).render("film", tmp_path, film_dpi=254, wedge=False)
    img = PILImage.open(out.paths[0])
    assert img.mode == "1"
    m = MARGIN_MM
    assert img.size == (round((20 + 2 * m) * 10), round((10 + 2 * m) * 10))
    white = np.asarray(img)
    row = round((5 + m) * 10)
    # Solid ink on the canvas's left half lands on the sheet's right half once mirrored.
    assert not white[row, 260:340].any()
    assert white[row, 160:250].all()  # right up to the cut: edge cells are clipped on film too


def test_film_step_wedge_is_screened_in_bottom_margin(tmp_path):
    out = _job(source=ht.Constant(0.0)).render("film", tmp_path, film_dpi=254)
    white = np.asarray(PILImage.open(out.paths[0]))
    m = MARGIN_MM
    band = white[round((10 + 3 + 2 + m) * 10) : round((10 + 3 + 6 + m) * 10)]
    assert (~band).mean() > 0.05


def test_cli_new_report_render(tmp_path, capsys):
    recipe = tmp_path / "job.json"
    assert main(["profiles"]) == 0
    assert main([
        "new", str(recipe), "--profile", "newsprint_nominal", "--size", "30x20", "--dpi", "100",
        "--ink", "red=#D6422B", "--ink", "blue=#1F3A63", "--overprint", "red+blue=#3A2036", "--seed", "3",
    ]) == 0
    assert main(["report", str(recipe), "--target", "film"]) == 0
    assert "newsprint_nominal" in capsys.readouterr().out
    assert main(["render", str(recipe), "--target", "svg", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "job.svg").exists()
    job = ht.Recipe.load(recipe)
    assert job.inks.overprint == {frozenset({"red", "blue"}): "#3A2036"}


def test_every_policy_decides_every_check_kind():
    """A kind no policy names defaults to "report" -- advisory even on film and pdf, which refuse.

    So a new check added without a policy decision would be silently toothless. Pin it.
    """
    for target, policy in POLICIES.items():
        assert set(policy) == set(CHECK_KINDS), f"{target} policy and CHECK_KINDS disagree"
        assert set(policy.values()) <= {"report", "warn", "cap", "refuse", "ignore"}


def test_check_kinds_lists_exactly_what_report_emits():
    """CHECK_KINDS is only useful if it tracks the Check() calls in recipe.py."""
    import ast
    import halftoner.recipe as recipe_mod

    tree = ast.parse(pathlib.Path(recipe_mod.__file__).read_text())
    emitted = {n.args[3].value for n in ast.walk(tree)
               if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Check"
               and len(n.args) >= 4 and isinstance(n.args[3], ast.Constant)}
    bare = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Check" and len(n.args) < 4]
    assert not bare, "every Check() must name its kind, not fall back to 'info'"
    assert emitted == set(CHECK_KINDS)
