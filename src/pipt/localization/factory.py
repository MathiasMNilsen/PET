"""Build a localization strategy from its config, by name.

The strategies are looked up in :data:`LOCALIZATIONS`, a table from the
config's ``name`` to a builder. Adding a strategy is one call to
:func:`register_localization`; nothing here needs editing.
"""

from typing import Callable, Union

import pandas as pd

from pipt.localization.common import normalize_parsed_info

__all__ = [
    "LOCALIZATIONS",
    "available_localizations",
    "build_localization_instance",
    "register_localization",
]


def _build_autoadaloc(*, info, rng=None, **_):
    from pipt.localization.auto_ada_loc import AutoAdaptiveLocalization
    return AutoAdaptiveLocalization(info, rng=rng)


def _build_localanalysis(*, info, data_indices, data_types, parameters, ensemble_size, **_):
    from pipt.localization.local_analysis import LocalAnalysisLocalization
    return LocalAnalysisLocalization(
        info=info,
        data_indices=data_indices,
        data_types=data_types,
        parameters=parameters,
        ensemble_size=ensemble_size,
    )


def _build_distance(*, info, data, parameters, ensemble_size, prior_info, **_):
    from pipt.localization.distance_localization import DistanceLocalization
    return DistanceLocalization(
        info=info,
        data=data,
        parameters=parameters,
        ensemble_size=ensemble_size,
        prior_info=prior_info,
    )


#: Config ``name`` -> builder. Every builder is called with the same keyword
#: arguments (``info`` plus everything :func:`build_localization_instance`
#: receives) and takes what it needs.
LOCALIZATIONS: dict[str, Callable[..., object]] = {
    "autoadaloc": _build_autoadaloc,
    "localanalysis": _build_localanalysis,
    "distance_loc": _build_distance,
}


def register_localization(name: str, builder: Callable[..., object], *, overwrite: bool = False) -> None:
    """Make a localization strategy selectable as ``localization = {name = ...}``.

    Parameters
    ----------
    name : str
        The value of the config's ``name`` key.
    builder : callable
        Called as ``builder(info=..., data_indices=..., data_types=...,
        parameters=..., ensemble_size=..., data=..., prior_info=..., rng=...)``; it may
        ignore what it does not need. Returns the strategy object, which the
        analyses use through its ``name`` attribute and by calling it.
    overwrite : bool, optional
        Allow replacing an existing entry. Off by default, so two packages
        claiming the same name is an error rather than a load-order lottery.
    """
    key = str(name).lower()
    if key in LOCALIZATIONS and not overwrite:
        raise ValueError(f"Localization {key!r} is already registered; pass overwrite=True to replace it.")
    LOCALIZATIONS[key] = builder


def available_localizations() -> list[str]:
    """The registered localization names, sorted."""
    return sorted(LOCALIZATIONS)


def build_localization_instance(
    parsed_info: Union[dict, list],
    data_indices: Union[list, None] = None,
    data_types: Union[list, None] = None,
    parameters: Union[list, None] = None,
    ensemble_size: Union[int, None] = None,
    data: Union[pd.DataFrame, None] = None,
    prior_info: Union[dict, None] = None,
    rng=None,
) -> object:
    """Create the localization strategy the config names.

    ``rng`` is the run's random stream, for strategies that draw (the
    auto-adaptive one shuffles the ensemble to estimate a noise level).
    """
    info = normalize_parsed_info(parsed_info)
    name = info.pop("name", None)
    if name is None:
        raise ValueError(f"Localization config has no 'name'; expected one of {available_localizations()}.")
    builder = LOCALIZATIONS.get(str(name).lower())
    if builder is None:
        raise ValueError(f"Unknown localization type {name!r}; expected one of {available_localizations()}.")
    return builder(
        info=info,
        data_indices=data_indices,
        data_types=data_types,
        parameters=parameters,
        ensemble_size=ensemble_size,
        data=data,
        prior_info=prior_info,
        rng=rng,
    )
