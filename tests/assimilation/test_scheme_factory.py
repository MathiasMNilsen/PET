"""Tests for the friendly scheme constructors.

The flavour is a parameter of the algorithm, not a different algorithm, so
``ESMDA(..., analysis="full")`` must resolve to exactly the class previously
named ``esmda_full``.
"""

import pytest

import pipt
from pipt.update_schemes import registry


ALGORITHMS = {
    "EnKF": ("enkf", ["approx", "full", "subspace"]),
    "ES": ("es", ["approx", "full", "subspace"]),
    "ESMDA": ("esmda", ["approx", "full", "subspace", "geo", "hybrid"]),
    "LMEnRML": ("lmenrml", ["approx", "full", "subspace"]),
    "GNEnRML": ("gnenrml", ["approx", "full", "subspace", "margis"]),
}


def test_top_level_exports():
    for name in ALGORITHMS:
        assert hasattr(pipt, name), f"pipt.{name} should be importable"
    assert hasattr(pipt, "build_scheme")


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_constructor_is_named_readably(name):
    assert getattr(pipt, name).__name__ == name


@pytest.mark.parametrize(
    "name,scheme,flavour",
    [(n, s, f) for n, (s, fs) in ALGORITHMS.items() for f in fs],
)
def test_every_flavour_documented_is_registered(name, scheme, flavour):
    """Each flavour named in a constructor's docstring must actually resolve."""
    assert registry.get_scheme(scheme, flavour) is not None
    assert flavour in getattr(pipt, name).__doc__


@pytest.mark.parametrize("name,scheme", [(n, s) for n, (s, _) in ALGORITHMS.items()])
def test_five_names_cover_all_eighteen_classes(name, scheme):
    """The five constructors between them reach every registered class."""
    flavours = [f for s, f in registry.available_schemes() if s == scheme]
    assert flavours, f"{scheme} has no registered flavours"


def test_constructors_collapse_the_name_explosion():
    total_classes = len(registry.available_schemes())
    assert total_classes == 18
    assert len(ALGORITHMS) == 5


def test_build_scheme_and_named_constructor_agree(monkeypatch):
    """Both paths must resolve to the same concrete class."""
    captured = {}

    class Spy:
        def __init__(self, da, en, sim):
            captured["args"] = (da, en, sim)

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "approx"), Spy)

    a = pipt.ESMDA({"d": 1}, {"e": 2}, "sim", analysis="approx")
    assert isinstance(a, Spy)
    assert captured["args"] == ({"d": 1}, {"e": 2}, "sim")

    b = pipt.build_scheme("esmda", {"d": 1}, {"e": 2}, "sim", analysis="approx")
    assert isinstance(b, Spy)


def test_default_analysis_is_approx(monkeypatch):
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "approx"), Spy)
    assert isinstance(pipt.ESMDA({}, {}, None), Spy)


def test_bad_flavour_reports_valid_ones():
    with pytest.raises(KeyError, match="no 'nope' analysis flavour"):
        pipt.ESMDA({}, {}, None, analysis="nope")


def test_concrete_classes_remain_importable():
    """The new layer is additive: old names still work for isinstance/subclassing."""
    from pipt.update_schemes import esmda_full, lmenrml_approx

    assert registry.get_scheme("esmda", "full") is esmda_full
    assert registry.get_scheme("lmenrml", "approx") is lmenrml_approx
