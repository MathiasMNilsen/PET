"""Lookup of analysis strategies by flavour name.

The scheme registry maps ``(scheme, analysis)`` to one class per combination,
because the flavour is currently baked into the class through mixin
composition. This maps the flavour *alone* to the strategy implementing it,
which is what a scheme needs once it takes ``analysis`` as a parameter and
holds the strategy rather than inheriting it.

Kept in its own module rather than in :mod:`pipt.update_schemes.analysis.base`:
the concrete flavours import the base, so a registry living there would import
its own importers. ``tests/test_import_hygiene.py`` guards the layering.
"""

from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update

__all__ = ["STRATEGIES", "available_strategies", "get_strategy", "register_strategy"]


#: Maps an ``analysis`` flavour to the strategy class implementing it.
STRATEGIES: dict[str, type] = {
    "approx": approx_update,
    "full": full_update,
    "subspace": subspace_update,
}


def register_strategy(analysis: str, cls: type, *, overwrite: bool = False) -> None:
    """Add a strategy, so out-of-tree flavours need not edit this file.

    Parameters
    ----------
    analysis : str
        Flavour name, as it appears in the config's ``analysis`` key.
    cls : type
        Strategy class implementing it.
    overwrite : bool, optional
        Allow replacing an existing entry. Defaults to ``False``, so two
        packages claiming one name is an error rather than a load-order
        lottery -- matching ``registry.register_scheme``.
    """
    key = str(analysis).lower()
    if key in STRATEGIES and not overwrite:
        raise ValueError(
            f"Analysis flavour '{key}' is already registered to "
            f"{STRATEGIES[key].__name__}; pass overwrite=True to replace it."
        )
    STRATEGIES[key] = cls


def available_strategies() -> list[str]:
    """Return the registered flavour names, sorted."""
    return sorted(STRATEGIES)


def get_strategy(analysis: str) -> type:
    """Look up the strategy class for a flavour.

    Raises
    ------
    KeyError
        If the flavour is not registered. The message lists the valid ones.
    """
    key = str(analysis).lower()
    if key in STRATEGIES:
        return STRATEGIES[key]
    raise KeyError(
        f"Unknown analysis flavour '{analysis}'. "
        f"Available flavours: {', '.join(available_strategies())}."
    )
