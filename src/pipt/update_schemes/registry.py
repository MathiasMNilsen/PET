"""Explicit registry of selectable assimilation schemes.

PIPT historically resolved a scheme by string surgery on the config::

    getattr(import_module('pipt.update_schemes.' + daalg[0]),
            f'{daalg[1]}_{analysis}')

That works, but it fails badly: a typo in ``daalg`` surfaces as a bare
``ModuleNotFoundError`` or ``AttributeError`` naming a symbol the user never
wrote, there is no way to ask what the valid combinations are, and any tool
wanting to list the available schemes has to guess at module contents.

This module replaces that with an explicit table built from real imports, in
the same spirit as ``pipt.localization.factory``. Lookup failures name the
offending key and list what is actually available.

Extending the registry
----------------------
Schemes living outside this repository -- for instance the private
``margIS_update`` implementation -- can register themselves without editing
this file::

    from pipt.update_schemes.registry import register_scheme
    register_scheme("myscheme", "approx", MySchemeApprox)
"""

from pipt.update_schemes.enkf import enkf_approx, enkf_full, enkf_subspace
from pipt.update_schemes.enrml import (
    gnenrml_approx,
    gnenrml_full,
    gnenrml_margis,
    gnenrml_subspace,
    lmenrml_approx,
    lmenrml_full,
    lmenrml_subspace,
)
from pipt.update_schemes.es import es_approx, es_full, es_subspace
from pipt.update_schemes.esmda import (
    esmda_approx,
    esmda_full,
    esmda_geo,
    esmda_subspace,
)
# esmda_hybrid is a multilevel variant and lives with the multilevel machinery.
from pipt.update_schemes.multilevel import esmda_hybrid

__all__ = [
    "SCHEMES",
    "available_schemes",
    "get_scheme",
    "register_scheme",
]


#: Maps ``(scheme, analysis)`` to the class implementing that combination.
#: The keys are exactly the two values a config supplies as ``daalg[1]`` and
#: ``analysis``; the class names are unchanged and remain importable directly.
SCHEMES: dict[tuple[str, str], type] = {
    ("enkf", "approx"): enkf_approx,
    ("enkf", "full"): enkf_full,
    ("enkf", "subspace"): enkf_subspace,
    ("es", "approx"): es_approx,
    ("es", "full"): es_full,
    ("es", "subspace"): es_subspace,
    ("esmda", "approx"): esmda_approx,
    ("esmda", "full"): esmda_full,
    ("esmda", "subspace"): esmda_subspace,
    ("esmda", "geo"): esmda_geo,
    ("esmda", "hybrid"): esmda_hybrid,
    ("lmenrml", "approx"): lmenrml_approx,
    ("lmenrml", "full"): lmenrml_full,
    ("lmenrml", "subspace"): lmenrml_subspace,
    ("gnenrml", "approx"): gnenrml_approx,
    ("gnenrml", "full"): gnenrml_full,
    ("gnenrml", "subspace"): gnenrml_subspace,
    # Backed by a private implementation when that package is installed, and by
    # an inert placeholder otherwise -- see enrml.py.
    ("gnenrml", "margis"): gnenrml_margis,
}


def register_scheme(scheme: str, analysis: str, cls: type, *, overwrite: bool = False) -> None:
    """Add a scheme to the registry.

    Parameters
    ----------
    scheme : str
        Scheme name, as it appears in ``daalg[1]``.
    analysis : str
        Analysis flavour, as it appears in ``analysis``.
    cls : type
        Class implementing the combination.
    overwrite : bool, optional
        Allow replacing an existing entry. Defaults to ``False`` so that two
        packages silently claiming the same key is an error rather than a
        load-order lottery.
    """
    key = (str(scheme).lower(), str(analysis).lower())
    if key in SCHEMES and not overwrite:
        raise ValueError(
            f"Scheme {key} is already registered to "
            f"{SCHEMES[key].__name__}; pass overwrite=True to replace it."
        )
    SCHEMES[key] = cls


def available_schemes() -> list[tuple[str, str]]:
    """Return the registered ``(scheme, analysis)`` combinations, sorted."""
    return sorted(SCHEMES)


def get_scheme(scheme: str, analysis: str) -> type:
    """Look up the class implementing a ``(scheme, analysis)`` combination.

    Raises
    ------
    KeyError
        If the combination is not registered. The message distinguishes an
        unknown scheme from a known scheme with an unsupported analysis
        flavour, and lists the valid options in both cases.
    """
    key = (str(scheme).lower(), str(analysis).lower())
    if key in SCHEMES:
        return SCHEMES[key]

    known = {name for name, _ in SCHEMES}
    if key[0] not in known:
        raise KeyError(
            f"Unknown assimilation scheme '{scheme}'. "
            f"Available schemes: {', '.join(sorted(known))}."
        )

    flavours = sorted(flavour for name, flavour in SCHEMES if name == key[0])
    raise KeyError(
        f"Scheme '{scheme}' has no '{analysis}' analysis flavour. "
        f"Available flavours for '{scheme}': {', '.join(flavours)}."
    )
