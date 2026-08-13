"""Analysis-step strategies for the assimilation schemes.

Currently exposes the shared strategy base. The concrete flavours
(``approx_update``, ``full_update``, ``subspace_update``) still live in
``pipt.update_schemes.update_methods_ns`` and are imported from there; they are
deliberately *not* re-exported here, because those modules import
``analysis.base`` and re-exporting them would make this package import itself.

They move into this package -- and become importable from here -- once the
schemes stop consuming them as mixins.
"""

from .base import AnalysisStrategy

__all__ = ["AnalysisStrategy"]
