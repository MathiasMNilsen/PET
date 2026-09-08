"""Localization strategies are selected from a table, and the table is open."""

import pytest

from pipt.localization import (
    LOCALIZATIONS,
    available_localizations,
    build_localization_instance,
    register_localization,
)


class Custom:
    name = "custom"

    def __init__(self, info, ensemble_size):
        self.info = info
        self.ensemble_size = ensemble_size


def _build_custom(*, info, ensemble_size, **_):
    return Custom(info, ensemble_size)


def test_the_shipped_strategies_are_registered():
    assert available_localizations() == ["autoadaloc", "distance_loc", "localanalysis"]


def test_a_registered_strategy_is_built_from_its_name(monkeypatch):
    monkeypatch.setitem(LOCALIZATIONS, "custom", _build_custom)

    loc = build_localization_instance({"name": "custom", "radius": 3}, None, None, None, 17)

    assert isinstance(loc, Custom)
    assert loc.info == {"radius": 3} and loc.ensemble_size == 17
    assert "custom" in available_localizations()


def test_registering_an_existing_name_needs_overwrite(monkeypatch):
    monkeypatch.setitem(LOCALIZATIONS, "custom", _build_custom)
    with pytest.raises(ValueError, match="already registered"):
        register_localization("custom", _build_custom)
    register_localization("custom", _build_custom, overwrite=True)


def test_register_localization_adds_to_the_table(monkeypatch):
    monkeypatch.delitem(LOCALIZATIONS, "brand_new", raising=False)
    register_localization("brand_new", _build_custom)
    try:
        assert "brand_new" in LOCALIZATIONS
    finally:
        LOCALIZATIONS.pop("brand_new", None)


def test_unknown_name_lists_what_is_available():
    with pytest.raises(ValueError, match="autoadaloc"):
        build_localization_instance({"name": "nope"}, None, None, None, 1)
