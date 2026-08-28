"""Shared base for the analysis-step analyses.

An *analysis* computes the state update for one assimilation
iteration. The three shipped flavours -- ``approx``, ``full`` and ``subspace``
-- differ only in how the ensemble-approximated sensitivity is inverted; they
share their calling convention and their linear-algebra helpers.

This is the PIPT counterpart to ``popt.optimization_methods.subroutines``:
small, focused numerical pieces the top-level scheme composes with, rather than
behaviour baked into the scheme's class name.

Historically these flavours were mixins combined into the scheme at class
definition time, producing a combinatorial explosion of names
(``esmda_approx``, ``esmda_full``, ``esmda_subspace``, ``lmenrml_approx``, ...).
Every algorithm class now takes ``analysis`` as a constructor argument and
binds the matching analysis instead (see ``AnalysisBindingMixin``). Mixing in still
works, for an analysis that genuinely cannot take this shape -- nothing shipped
here needs it any more, now that ``margis`` binds like the rest -- but doing
so is riskier than it looks: see ``AnalysisBindingMixin``'s module docstring for why
the scheme base usually has to be listed first, and what that can do to
method resolution.

Analysis contract
-----------------
``update(enX, enY, enE, **kwargs) -> np.ndarray | None``
    Return the state update step, shape ``(nx, ne)``, or ``None`` if the
    analysis delivers its result by assignment onto the scheme instead (see
    below).

Analyses reach everything they need through ``self.scheme``: the damping
parameter ``self.scheme.lam``, ``self.scheme.trunc_energy``,
``self.scheme.localization``, ``self.scheme.prior_enX``,
``self.scheme.cov_data``, and so on. Some of those are the scheme's own
attributes and some belong to its ensemble, but the scheme exposes both as
properties (see :class:`~pipt.update_schemes.core.AssimilationScheme`),
so an analysis never has to know which -- and there is no forwarding
machinery on this side at all. A new flavour that needs a value no existing
one uses just reads ``self.scheme.<name>``; if the scheme does not already
expose it, adding one property there is the whole change.

``self.scheme`` resolves for both ways an analysis can be used:

- **Bound** -- ``self.scheme`` is the scheme it was constructed against.
- **Mixed in** -- ``self`` *is* the scheme, so ``self.scheme`` is ``self``
  (see :attr:`scheme` below). Nothing shipped here still needs this
  (``margis`` binds like the rest now); it remains supported for an analysis
  whose calling convention genuinely does not fit the bound shape.

An analysis that delivers its result by assignment (``subspace_update`` sets
``w_step``; ``full_update`` caches ``Am``) writes it onto ``self.scheme``
explicitly, the same way it reads -- e.g. ``self.scheme.w_step = ...`` --
not onto ``self``. There is nothing that forwards a plain ``self.w_step =
...`` for you; an analysis that wrote to itself here would have the scheme's
``hasattr(self, 'w_step')`` silently stay False, no error.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import solve as _dense_solve
from scipy.linalg import sqrtm as _dense_sqrtm

__all__ = ["AnalysisBase"]


class AnalysisBase(ABC):
    """Base class for analysis-step analyses.

    Provides the linear-algebra helpers every flavour needs. Both accept either
    a full 2-D matrix or a 1-D array holding just the diagonal, which is how
    PIPT represents a diagonal data covariance without materialising ``nd x nd``
    zeros.

    Two usages
    ----------
    **Bound** (what every algorithm class does, for every flavour in its
    ``COMPATIBLE_ANALYSES``) -- constructed against a scheme it holds a
    reference to::

        strategy = approx_update(scheme)
        step = strategy.update(enX, enY, enE)

    which is what lets ``analysis`` be a constructor argument of one scheme
    class rather than picking which of several classes you get. Inside
    ``update()``, context is read explicitly off ``self.scheme`` -- there is
    no delegation step to run first; ``self.scheme`` is just the object
    passed to the constructor, and it exposes ensemble state as properties
    of its own.

    **Mixed in** -- nothing shipped here still needs this (``margis`` binds
    like the rest now); it remains supported for an analysis whose calling
    convention genuinely does not fit the bound shape above::

        class some_scheme(SomeAlgorithm, some_analysis): ...

    ``self`` *is* the scheme here, so ``self.scheme`` (the :attr:`scheme`
    property below) simply returns ``self`` -- ``self.scheme.lam`` and
    ``self.lam`` are then the same read, resolved by ordinary inheritance.

    An unbound, un-mixed-in analysis has ``self.scheme`` fall back to
    ``self`` too, so a context read raises a plain ``AttributeError`` rather
    than finding a half-initialised scheme.
    """

    def __init__(self, scheme=None):
        """
        Parameters
        ----------
        scheme : object, optional
            Scheme this analysis computes updates for. ``None`` leaves the
            analysis unbound. Never invoked in the mixin case: no
            ``__init__`` in that MRO chains to ``super()``.
        """
        self._scheme = scheme

    @property
    def scheme(self):
        """The scheme to read context from and write results onto.

        The bound value if there is one; otherwise ``self`` -- which is
        exactly right when *mixed in* (``self`` already is the scheme, so
        ``self.scheme.x`` and ``self.x`` are the same read) and merely
        produces a plain ``AttributeError`` from an unbound, un-mixed-in
        analysis rather than a special-cased error path.
        """
        bound = getattr(self, "_scheme", None)
        return bound if bound is not None else self

    @abstractmethod
    def update(self, enX, enY, enE, **kwargs):
        """Compute the analysis update step.

        Parameters
        ----------
        enX : np.ndarray
            State ensemble matrix, shape ``(nx, ne)``.
        enY : np.ndarray
            Predicted data ensemble matrix, shape ``(nd, ne)``.
        enE : np.ndarray
            Perturbed observation ensemble, shape ``(nd, ne)``.
        **kwargs
            Analysis-specific extras, e.g. ``prior`` or ``enAdj``.

        Returns
        -------
        np.ndarray or None
            State update step, shape ``(nx, ne)``.
        """

    @staticmethod
    def solve(A, B):
        """Apply ``A⁻¹ B``, supporting both matrix (2-D) and diagonal (1-D) ``A``.

        ``np.ndim`` is used rather than ``A.ndim`` so that plain lists and
        scalars -- which a covariance can still be when it comes straight from a
        config file -- are handled instead of raising ``AttributeError``.
        """
        if np.ndim(A) == 2:
            return _dense_solve(A, B)
        return (np.asarray(A) ** (-1))[:, None] * B

    @staticmethod
    def sqrtm(A):
        """Matrix square root, supporting both matrix and diagonal inputs."""
        if np.ndim(A) == 2:
            return _dense_sqrtm(A)
        return np.sqrt(A)
