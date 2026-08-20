"""Holding an analysis strategy rather than inheriting one.

Lets a scheme take its analysis flavour as an argument, so one class covers
``approx``/``full``/``subspace`` instead of one class per combination.

How a scheme ends up paired with a strategy
--------------------------------------------
Every scheme mixing in :class:`StrategyMixin` declares, right on the class,
which flavours it supports and which class handles each -- e.g.
``esmda.py``::

    class ESMDA(StrategyMixin, ...):
        COMPATIBLE_ANALYSES = {
            "approx": approx_update,
            "full": full_update,
            "subspace": subspace_update,
        }

Walked through for ``ESMDA(da, en, sim, analysis="approx")``, at construction
time::

    1. ESMDA.__init__(...)            [pipt/update_schemes/esmda.py]
           |
           |  self.bind_strategy(self.resolve_analysis(analysis, keys_da))
           v
    2. resolve_analysis("approx", keys_da) -> "approx"     [this module]
           picks the flavour: explicit argument, else keys_da["analysis"],
           else "approx".
           |
           v
    3. bind_strategy("approx")                              [this module]
           looks "approx" up in `self.COMPATIBLE_ANALYSES`, giving
           approx_update. self.strategy = approx_update(self) -- an
           *instance*, holding a reference back to the scheme (`self`) it
           was built from.

    Later, once per iteration:

    4. ESMDA.calc_analysis() calls self.update(enX=..., enY=..., ...)
           |
           |  StrategyMixin.update() just forwards:
           v
       self.strategy.update(enX=..., enY=..., ...)           [analysis/approx.py]
           does the actual linear algebra. It reads things like `self.lam`
           and `self.trunc_energy` -- `self` here is the *strategy*, but
           AnalysisStrategy.__getattr__ (analysis/base.py) forwards any
           attribute it does not have itself to the scheme it was bound to
           in step 3. So `self.lam` inside the strategy is really
           `esmda_instance.lam`.

``EnKF``/``ES`` never revisit a data group, so the prior-increment term
``full`` adds over ``approx`` never applies -- the two produce identical
output (pinned by the characterisation suite). Rather than special-casing
that in code, ``EnKF.COMPATIBLE_ANALYSES`` just points ``"full"`` at the same
class as ``"approx"``:

    COMPATIBLE_ANALYSES = {"approx": approx_update, "full": approx_update, "subspace": subspace_update}

``ES`` inherits this dict unchanged, so the fact lives in exactly one place
and applies regardless of how the scheme was constructed.

Mixing in is still supported, but nothing live uses it
--------------------------------------------------------------------------
``bind_strategy`` still checks whether a strategy was mixed directly into
the scheme's bases (``_flavour_is_mixed_in``) and, if so, leaves
``self.strategy`` unset and lets that inherited ``update()`` take over
instead of building one. Both flavours that used to need this --
``hybrid_update`` (multilevel ES-MDA) and ``margIS_update`` (marg-IS) -- now
bind normally instead: both take the same ``(enX, enY, enE, **kwargs)``
shape as ``approx_update`` and friends, so ``esmda_hybrid.COMPATIBLE_ANALYSES
= {"hybrid": hybrid_update}`` and ``GNEnRML.COMPATIBLE_ANALYSES["margis"] =
margIS_update`` bind them the normal way.

Mixing a strategy directly into a scheme's bases is riskier than it looks
when the scheme base is listed first, which it usually must be: whichever
class the scheme's own ``__init__`` needs to resolve to has to come first,
but that can leave the *strategy's* ``update()`` shadowed by
``StrategyMixin.update()`` -- found first via the scheme's own MRO chain --
regardless of what ``bind_strategy`` decides. That bit both ``esmda_hybrid``
and ``gnenrml_margis`` (the latter fixed with an explicit ``update``
override before margis was converted to bind normally; see the CHANGELOG).
The one class still doing this is ``co_lm_enrml`` (``pipt.update_schemes.
enrml``) -- kept in the source but never constructed, so the risk is inert.
Prefer binding (a ``COMPATIBLE_ANALYSES`` entry) over mixing in for any new
flavour that fits the ``(enX, enY, enE, **kwargs)`` shape; mixing in is only
for a strategy that genuinely cannot, the way ``margIS_update`` used to.
"""

__all__ = ["StrategyMixin"]


class StrategyMixin:
    """Resolve an analysis flavour to a strategy object and delegate to it."""

    #: Flavour name -> strategy class to build with ``self`` as its scheme.
    #: Every scheme mixing this in sets its own (see module docstring). A
    #: scheme that instead gets a flavour by mixing the strategy directly
    #: into its bases needs no entry for it here, since ``bind_strategy``
    #: never consults this dict in that case.
    COMPATIBLE_ANALYSES: dict[str, type] = {}

    #: Bound strategy, or ``None`` when the flavour is supplied by a mixin.
    strategy = None

    def resolve_analysis(self, analysis=None, keys_da=None) -> str:
        """Decide the flavour: explicit argument, else the config, else "approx"."""
        if analysis is not None:
            return str(analysis).lower()
        if keys_da is not None:
            return str(keys_da.get("analysis", "approx")).lower()
        return "approx"

    def bind_strategy(self, analysis) -> None:
        """Bind the strategy for ``analysis``, unless a mixin already supplies one.

        Nothing shipped in this repository takes that path today (see the
        module docstring); it remains for a scheme that mixes a strategy
        directly into its bases instead of listing it in
        ``COMPATIBLE_ANALYSES``, in which case it keeps the inherited
        implementation and binds nothing.
        """
        self.analysis = analysis
        if self._flavour_is_mixed_in():
            self.strategy = None
            return
        if analysis not in self.COMPATIBLE_ANALYSES:
            raise KeyError(
                f"{type(self).__name__} has no {analysis!r} analysis flavour. "
                f"Available: {', '.join(sorted(self.COMPATIBLE_ANALYSES))}."
            )
        self.strategy = self.COMPATIBLE_ANALYSES[analysis](self)

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
