"""Holding an analysis strategy rather than inheriting one.

Lets a scheme take its analysis flavour as an argument, so one class covers
``approx``/``full``/``subspace`` instead of one class per combination.

Why this is a mixin and not part of
:class:`~pipt.update_schemes.core.AssimilationSchemeBase`: in the legacy
class layout the base *precedes* the strategy in the MRO::

    esmda_approx -> esmdaMixIn -> ... -> AssimilationSchemeBase -> approx_update

so an ``update()`` defined on the base would shadow the mixed-in flavour's for
every one of those classes. Keeping the delegate here means only classes that
opt in are affected.
"""

from pipt.update_schemes.analysis.registry import get_strategy

__all__ = ["StrategyMixin"]


class StrategyMixin:
    """Resolve an analysis flavour to a strategy object and delegate to it."""

    #: Set on a subclass to pin its flavour, which is how the historical
    #: per-flavour names (``esmda_approx`` and friends) stay meaningful.
    #: ``None`` means take the flavour from the argument or the config.
    FLAVOUR: str | None = None

    #: Bound strategy, or ``None`` when the flavour is supplied by a mixin.
    strategy = None

    def resolve_analysis(self, analysis=None, keys_da=None) -> str:
        """Decide the flavour: pinned by the class, then argument, then config."""
        if self.FLAVOUR is not None:
            return self.FLAVOUR
        if analysis is not None:
            return str(analysis).lower()
        if keys_da is not None:
            return str(keys_da.get("analysis", "approx")).lower()
        return "approx"

    def bind_strategy(self, analysis) -> None:
        """Bind the strategy for ``analysis``, unless a mixin already supplies one.

        ``esmda_hybrid`` and ``gnenrml_margis`` get their ``update`` by mixing
        in ``hybrid_update`` / ``margIS_update``, neither of which is registered
        as a flavour or even derives from ``AnalysisStrategy``. Those keep the
        inherited implementation and bind nothing.
        """
        self.analysis = analysis
        self.strategy = None if self._flavour_is_mixed_in() else get_strategy(analysis)(self)

    def _flavour_is_mixed_in(self) -> bool:
        """True if some other class in the MRO already defines ``update``."""
        return any(
            "update" in klass.__dict__
            for klass in type(self).__mro__
            if klass is not StrategyMixin
        )

    def update(self, *args, **kwargs):
        """Delegate the analysis step to the bound strategy.

        Only reached when nothing else in the MRO defines ``update``; a
        mixed-in flavour takes precedence and never gets here.
        """
        if self.strategy is None:
            raise AttributeError(
                f"{type(self).__name__} has no analysis strategy bound and no "
                f"mixed-in update(); bind_strategy() was not called."
            )
        return self.strategy.update(*args, **kwargs)
