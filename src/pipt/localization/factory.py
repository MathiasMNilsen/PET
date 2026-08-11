"""Factory helpers for localization strategy selection."""

from typing import Union

import pandas as pd

from pipt.localization.common import normalize_parsed_info


__all__ = ["build_localization_instance"]


def build_localization_instance(
    parsed_info: Union[dict, list],
    data_indices: Union[list, None] = None,
    data_types: Union[list, None] = None,
    parameters: Union[list, None] = None,
    ensemble_size: Union[int, None] = None,
    data: Union[pd.DataFrame, None] = None,
    prior_info: Union[dict, None] = None,
) -> object:
    
    """Create localization strategy instance matching configured mode."""
    from pipt.localization.auto_ada_loc import AutoAdaptiveLocalization
    from pipt.localization.distance_localization import DistanceLocalization
    from pipt.localization.local_analysis import LocalAnalysisLocalization

    info = normalize_parsed_info(parsed_info)
    loc_type = info.pop("name", None)

    if loc_type is not None:
        if loc_type == "autoadaloc":
            return AutoAdaptiveLocalization(info)

        if loc_type == "localanalysis":
            return LocalAnalysisLocalization(
                info=info,
                data_indices=data_indices,
                data_types=data_types,
                parameters=parameters,
                ensemble_size=ensemble_size,
            )
        if loc_type == "distance_loc":
            return DistanceLocalization(
                info=info,
                data=data,
                parameters=parameters,
                ensemble_size=ensemble_size,
                prior_info=prior_info,
            )
    else:
        raise ValueError(f"Unknown localization type: {loc_type}")


