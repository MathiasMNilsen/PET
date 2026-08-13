"""Tests for the explicit scheme registry and init_da dispatch.

Also pins the public scheme class names, which are imported directly by user
code and must therefore keep working.
"""

import pytest

from pipt import pipt_init
from pipt.update_schemes import registry


# ----------------------------------------------------------------------
# Public class names are API
# ----------------------------------------------------------------------

PUBLIC_SCHEME_NAMES = [
    "enkf_approx", "enkf_full", "enkf_subspace",
    "es_approx", "es_full", "es_subspace",
    "esmda_approx", "esmda_full", "esmda_subspace", "esmda_geo", "esmda_hybrid",
    "lmenrml_approx", "lmenrml_full", "lmenrml_subspace",
    "gnenrml_approx", "gnenrml_full", "gnenrml_subspace", "gnenrml_margis",
]


@pytest.mark.parametrize("name", PUBLIC_SCHEME_NAMES)
def test_scheme_name_importable_from_package(name):
    """User code does `from pipt.update_schemes import lmenrml_approx`."""
    import pipt.update_schemes as us

    assert hasattr(us, name), f"{name} is public API and must stay importable"


def test_co_lm_enrml_kept_but_inactive():
    """Retained in the source and importable, but not star-exported or selectable."""
    import pipt.update_schemes as us
    from pipt.update_schemes.enrml import co_lm_enrml

    assert co_lm_enrml is not None
    assert not hasattr(us, "co_lm_enrml"), "co_lm_enrml should stay out of the star-export"
    assert not any(cls is co_lm_enrml for cls in registry.SCHEMES.values())


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

def test_registry_covers_every_public_name():
    registered = {cls.__name__ for cls in registry.SCHEMES.values()}
    assert registered == set(PUBLIC_SCHEME_NAMES)


def test_get_scheme_resolves_and_is_case_insensitive():
    from pipt.update_schemes import esmda_approx

    assert registry.get_scheme("esmda", "approx") is esmda_approx
    assert registry.get_scheme("ESMDA", "Approx") is esmda_approx


def test_available_schemes_is_sorted_pairs():
    combos = registry.available_schemes()
    assert combos == sorted(combos)
    assert ("esmda", "geo") in combos


def test_unknown_scheme_error_lists_alternatives():
    with pytest.raises(KeyError, match="Unknown assimilation scheme") as err:
        registry.get_scheme("esmdaa", "approx")
    assert "esmda" in str(err.value)


def test_unknown_flavour_error_is_distinct_and_lists_flavours():
    with pytest.raises(KeyError, match="no 'banana' analysis flavour") as err:
        registry.get_scheme("esmda", "banana")
    message = str(err.value)
    assert "geo" in message and "approx" in message


def test_register_scheme_roundtrip():
    class Dummy:
        pass

    registry.register_scheme("dummy", "approx", Dummy)
    try:
        assert registry.get_scheme("dummy", "approx") is Dummy
        with pytest.raises(ValueError, match="already registered"):
            registry.register_scheme("dummy", "approx", Dummy)
        registry.register_scheme("dummy", "approx", Dummy, overwrite=True)
    finally:
        registry.SCHEMES.pop(("dummy", "approx"), None)


# ----------------------------------------------------------------------
# init_da validation
# ----------------------------------------------------------------------

def test_init_da_missing_daalg():
    with pytest.raises(ValueError, match="DAALG is missing"):
        pipt_init.init_da({}, {}, None)


def test_init_da_malformed_daalg():
    with pytest.raises(ValueError, match="both the assimilation type"):
        pipt_init.init_da({"daalg": ["esmda"]}, {}, None)


def test_init_da_missing_analysis():
    with pytest.raises(ValueError, match="ANALYSIS is missing"):
        pipt_init.init_da({"daalg": ["esmda", "esmda"]}, {}, None)


def test_init_da_unknown_scheme_reports_clearly():
    """The old importlib path raised a bare ModuleNotFoundError here."""
    with pytest.raises(KeyError, match="Unknown assimilation scheme"):
        pipt_init.init_da(
            {"daalg": ["nope", "nope"], "analysis": "approx"}, {}, None
        )
