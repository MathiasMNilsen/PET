"""Friendly constructors for the assimilation schemes.

PIPT names a scheme by concatenating the algorithm with its analysis flavour,
which produces one class per combination: ``esmda_approx``, ``esmda_full``,
``esmda_subspace``, ``esmda_geo``, ``esmda_hybrid``, ``lmenrml_approx``, and so
on -- eighteen names for five algorithms.

The flavour is a *parameter* of the algorithm, not a different algorithm, so
this module exposes one constructor per algorithm and takes the flavour as an
argument::

    from pipt import ESMDA
    scheme = ESMDA(cfg_da, cfg_en, sim, analysis="approx")

This mirrors how ``popt`` exposes ``EnOpt``/``LineSearch``/``TrustRegion`` as
one name per algorithm. The underlying concrete classes are unchanged and stay
importable, so ``isinstance`` checks and subclassing still work; these
constructors resolve through :mod:`pipt.update_schemes.registry` and return an
instance of exactly the same class as before.
"""

from pipt.update_schemes.registry import get_scheme

__all__ = ["EnKF", "ES", "ESMDA", "LMEnRML", "GNEnRML", "build_scheme"]


def build_scheme(scheme, da_input, en_input, sim, analysis="approx"):
    """Construct any registered scheme by name.

    Parameters
    ----------
    scheme : str
        Algorithm name, e.g. ``"esmda"``.
    da_input : dict
        Parsed data-assimilation config.
    en_input : dict
        Parsed ensemble config.
    sim : object
        Forward simulator instance.
    analysis : str, optional
        Analysis flavour. Defaults to ``"approx"``.

    Returns
    -------
    object
        The instantiated scheme.
    """
    return get_scheme(scheme, analysis)(da_input, en_input, sim)


def _make(scheme, flavours, doc_summary):
    """Build a named constructor for one algorithm."""

    def constructor(da_input, en_input, sim, analysis="approx"):
        return build_scheme(scheme, da_input, en_input, sim, analysis=analysis)

    constructor.__name__ = scheme
    constructor.__qualname__ = scheme
    constructor.__doc__ = f"""{doc_summary}

    Parameters
    ----------
    da_input : dict
        Parsed data-assimilation config.
    en_input : dict
        Parsed ensemble config.
    sim : object
        Forward simulator instance.
    analysis : str, optional
        Analysis flavour, one of: {', '.join(repr(f) for f in flavours)}.
        Defaults to ``'approx'``.

    Returns
    -------
    object
        Instance of the concrete ``{scheme}_<analysis>`` class.
    """
    return constructor


EnKF = _make(
    "enkf",
    ("approx", "full", "subspace"),
    "Ensemble Kalman Filter.",
)
EnKF.__name__ = EnKF.__qualname__ = "EnKF"

ES = _make(
    "es",
    ("approx", "full", "subspace"),
    "Ensemble Smoother.",
)
ES.__name__ = ES.__qualname__ = "ES"

ESMDA = _make(
    "esmda",
    ("approx", "full", "subspace", "geo", "hybrid"),
    "Ensemble Smoother with Multiple Data Assimilation.",
)
ESMDA.__name__ = ESMDA.__qualname__ = "ESMDA"

LMEnRML = _make(
    "lmenrml",
    ("approx", "full", "subspace"),
    "Levenberg-Marquardt Ensemble Randomized Maximum Likelihood.",
)
LMEnRML.__name__ = LMEnRML.__qualname__ = "LMEnRML"

GNEnRML = _make(
    "gnenrml",
    ("approx", "full", "subspace", "margis"),
    "Gauss-Newton Ensemble Randomized Maximum Likelihood.",
)
GNEnRML.__name__ = GNEnRML.__qualname__ = "GNEnRML"
