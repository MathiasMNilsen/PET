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
    """Every advertised (scheme, flavour) pair must still resolve to a class."""
    assert registry.get_scheme(scheme, flavour) is not None


@pytest.mark.parametrize("name,scheme", [(n, s) for n, (s, _) in ALGORITHMS.items()])
def test_five_names_cover_all_eighteen_classes(name, scheme):
    """The five constructors between them reach every registered class."""
    flavours = [f for s, f in registry.available_schemes() if s == scheme]
    assert flavours, f"{scheme} has no registered flavours"


def test_constructors_collapse_the_name_explosion():
    total_classes = len(registry.available_schemes())
    assert total_classes == 18
    assert len(ALGORITHMS) == 5


def test_build_scheme_still_dispatches_through_the_registry(monkeypatch):
    """`build_scheme` resolves by name; the classes no longer do.

    `pipt.ESMDA` used to be a function that looked the flavour up in the
    registry. It is now the class itself, so only the name-driven entry points
    -- build_scheme and init_da -- consult the registry.
    """
    captured = {}

    class Spy:
        def __init__(self, da, en, sim):
            captured["args"] = (da, en, sim)

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "approx"), Spy)

    b = pipt.build_scheme("esmda", {"d": 1}, {"e": 2}, "sim", analysis="approx")
    assert isinstance(b, Spy)
    assert captured["args"] == ({"d": 1}, {"e": 2}, "sim")


def test_registry_aliases_pin_the_flavour_the_class_name_promises():
    """`esmda_full` must still mean "full", now via FLAVOUR rather than a mixin."""
    for scheme, flavour in registry.available_schemes():
        cls = registry.get_scheme(scheme, flavour)
        pinned = getattr(cls, "FLAVOUR", None)
        if pinned is not None:
            # es_full/enkf_full historically resolved to the approx strategy,
            # because neither scheme iterates.
            assert pinned in {flavour, "approx"}, (
                f"{cls.__name__} pins {pinned!r} but is registered under {flavour!r}"
            )


def test_geo_and_hybrid_stay_separate_classes():
    """Not every registered flavour is a strategy.

    `geo` and `hybrid` are distinct algorithms sharing the ESMDA name, so they
    remain their own classes and are reachable through the registry rather than
    through `ESMDA(analysis=...)`.
    """
    from pipt.update_schemes.analysis.registry import available_strategies

    assert "geo" not in available_strategies()
    assert "hybrid" not in available_strategies()
    assert registry.get_scheme("esmda", "geo") is not None
    assert registry.get_scheme("esmda", "hybrid") is not None


def test_default_analysis_is_approx(monkeypatch):
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "approx"), Spy)
    assert isinstance(pipt.build_scheme("esmda", {}, {}, None), Spy)


def test_bad_flavour_reports_valid_ones():
    with pytest.raises(KeyError, match="no 'nope' analysis flavour"):
        pipt.build_scheme("esmda", {}, {}, None, analysis="nope")


def test_concrete_classes_remain_importable():
    """The new layer is additive: old names still work for isinstance/subclassing."""
    from pipt.update_schemes import esmda_full, lmenrml_approx

    assert registry.get_scheme("esmda", "full") is esmda_full
    assert registry.get_scheme("lmenrml", "approx") is lmenrml_approx


def test_factory_honours_config_analysis():
    """The factory must not silently disagree with init_da.

    `analysis` used to default to "approx" in the factory while init_da read it
    from the config, so a config asking for "subspace" built esmda_approx
    through one entry point and esmda_subspace through the other.
    """
    import inspect

    from pipt import ESMDA, build_scheme

    # The defaults are what caused the disagreement: "approx" here vs the
    # config's value in init_da.
    assert inspect.signature(ESMDA).parameters["analysis"].default is None
    assert inspect.signature(build_scheme).parameters["analysis"].default is None


def test_factory_resolves_each_flavour_from_config():
    from pipt.update_schemes.registry import get_scheme

    for flavour in ("approx", "full", "subspace"):
        cfg_da = {"scheme": "esmda", "analysis": flavour}
        assert get_scheme(cfg_da["scheme"], cfg_da["analysis"]) is get_scheme(
            "esmda", flavour
        )


def test_config_analysis_beats_the_fallback(monkeypatch):
    """A config asking for a flavour must not be overridden by the default."""
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "subspace"), Spy)
    cfg = {"scheme": "esmda", "analysis": "subspace"}
    assert isinstance(pipt.build_scheme("esmda", cfg, {}, None), Spy)


def test_explicit_analysis_beats_the_config(monkeypatch):
    class Spy:
        def __init__(self, da, en, sim):
            pass

    monkeypatch.setitem(registry.SCHEMES, ("esmda", "full"), Spy)
    cfg = {"scheme": "esmda", "analysis": "subspace"}
    assert isinstance(pipt.build_scheme("esmda", cfg, {}, None, analysis="full"), Spy)


def test_class_resolves_flavour_by_the_same_precedence():
    """The classes apply explicit -> config -> approx, as build_scheme does."""
    from pipt.update_schemes.esmda import ESMDA, esmda_subspace

    resolve = ESMDA.resolve_analysis
    assert resolve(ESMDA, "full", {"analysis": "subspace"}) == "full"
    assert resolve(ESMDA, None, {"analysis": "subspace"}) == "subspace"
    assert resolve(ESMDA, None, {}) == "approx"
    # A pinned alias ignores both, because its name is the promise.
    assert resolve(esmda_subspace, "full", {"analysis": "approx"}) == "subspace"
