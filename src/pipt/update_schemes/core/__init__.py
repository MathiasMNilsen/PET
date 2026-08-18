"""Machinery every assimilation scheme is built from.

Separated from the algorithms themselves so that ``pipt.update_schemes`` reads
as a list of schemes rather than a mixture of schemes and the scaffolding they
stand on. Three pieces, composed in this order by each scheme::

    class ESMDA(AssimilationWorkflowMixin, StrategyMixin, AssimilationSchemeBase)

:class:`AssimilationSchemeBase`
    The iteration loop, convergence bookkeeping, restart handling and the
    result object. Subclasses supply :meth:`~AssimilationSchemeBase.update_step`.
:class:`StrategyMixin`
    Resolves the ``analysis`` flavour to a strategy object and delegates
    ``update()`` to it, so the flavour is a parameter rather than part of the
    class name.
:class:`AssimilationWorkflowMixin`
    Diagnostics, artifact saving and outlier handling, expressed through the
    hooks the loop calls. A scheme wanting none of it simply does not mix it in.
"""

from .scheme_base import AssimilationResult, AssimilationSchemeBase
from .strategy import StrategyMixin
from .workflow import AssimilationWorkflowMixin

__all__ = [
    "AssimilationSchemeBase",
    "AssimilationResult",
    "StrategyMixin",
    "AssimilationWorkflowMixin",
]
