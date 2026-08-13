"""Inversion (estimation, data assimilation)

--8<-- "pipt/README.md"
"""

__pdoc__ = {}

# import replacement module and override submodule in pipt
# import sys
# from #replacement_package import #replacement_submodule
# sys.modules["pipt.#module.#submodule"] = #replacement_submodule

from pipt.update_schemes.factory import (  # noqa: E402
    ES,
    ESMDA,
    EnKF,
    GNEnRML,
    LMEnRML,
    build_scheme,
)
from pipt.update_schemes.registry import (  # noqa: E402
    available_schemes,
    get_scheme,
    register_scheme,
)

__all__ = [
    "EnKF",
    "ES",
    "ESMDA",
    "LMEnRML",
    "GNEnRML",
    "build_scheme",
    "available_schemes",
    "get_scheme",
    "register_scheme",
]
