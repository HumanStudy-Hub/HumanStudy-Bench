"""Numeric evidence must preserve exponent signs, including documented PDF text loss."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("quote_effects", Path(__file__).parents[1] / "scripts/research/effects_core.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_tiny_p_values_are_not_zero():
    assert core.quoted(1.9e-11, ["0, 5.3e-11"]) is None
    assert core.quoted(1.9e-11, ["1.9e−11"]) == "exact"


def test_lossy_exponents_require_page_and_review_note():
    record = {"quote": "p = 1.9e 11", "locator": {"page": 2}, "note": "rendered page checked"}
    assert core.rendered_quote(1.9e-11, record)
    assert not core.rendered_quote(1.9e-11, {**record, "note": ""})
    assert not core.rendered_quote(1.9e-11, {**record, "locator": {}})
    assert not core.rendered_quote(1.9e-11, {**record, "quote": "p = 1.9e+11"})


def test_separated_minus_is_joined_only_for_a_true_minus_where_a_number_starts():
    record = {"locator": {"page": 2}, "note": "rendered page checked"}
    for value, quote in [(-0.40, "95% CI [0.25 – 0.40]"), (-0.40, "95% CI [0.25 − 0.40]"),
                         (-5, "Study 2 - 5 items"), (-3.0, "Table 1 – 3.0"), (-0.011, "b = - 0.011")]:
        assert not core.rendered_quote(value, {**record, "quote": quote}), quote
    for value, quote in [(-0.011, "b = − 0.011"), (-0.5, "(− 0.5)"), (-0.4, "95% CI [0.1, − 0.4]")]:
        assert core.rendered_quote(value, {**record, "quote": quote}), quote
