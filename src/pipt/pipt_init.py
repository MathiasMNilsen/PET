"""Entry point for constructing an assimilation scheme from parsed config."""

from pipt.update_schemes.registry import get_scheme

__all__ = ["init_da"]


def init_da(da_input, en_input, sim):
    """Build the assimilation scheme object described by the config.

    Parameters
    ----------
    da_input : dict
        Parsed ``dataassim`` section. Must contain ``daalg`` as a two-element
        sequence ``[family, scheme]`` and, unless the scheme has a single
        flavour, ``analysis``.
    en_input : dict
        Parsed ``ensemble`` section.
    sim : object
        Forward simulator instance.

    Returns
    -------
    object
        Instantiated scheme.

    Raises
    ------
    ValueError
        If ``daalg`` is missing or malformed.
    KeyError
        If the requested scheme/analysis combination is not registered. The
        message lists the valid options.
    """
    daalg = da_input.get("daalg")
    if daalg is None:
        raise ValueError("DAALG is missing from the data-assimilation config.")
    if not isinstance(daalg, (list, tuple)) or len(daalg) != 2:
        raise ValueError(
            "DAALG must give both the assimilation type and the update method, "
            f"e.g. ['esmda', 'esmda']; got {daalg!r}."
        )

    analysis = da_input.get("analysis")
    if analysis is None:
        raise ValueError(
            f"ANALYSIS is missing from the data-assimilation config. "
            f"It selects the analysis flavour for scheme '{daalg[1]}'."
        )

    scheme_cls = get_scheme(daalg[1], analysis)
    return scheme_cls(da_input, en_input, sim)
