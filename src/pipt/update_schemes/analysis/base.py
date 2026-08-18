"""Shared base for the analysis-step strategies.

An *analysis strategy* computes the state update for one assimilation
iteration. The three shipped flavours -- ``approx``, ``full`` and ``subspace``
-- differ only in how the ensemble-approximated sensitivity is inverted; they
share their calling convention and their linear-algebra helpers.

This is the PIPT counterpart to ``popt.optimization_methods.subroutines``:
small, focused numerical pieces the top-level scheme composes with, rather than
behaviour baked into the scheme's class name.

Historically these flavours were mixins combined into the scheme at class
definition time, producing a combinatorial explosion of names
(``esmda_approx``, ``esmda_full``, ``esmda_subspace``, ``lmenrml_approx``, ...).
They remain usable as mixins -- every existing scheme still works unchanged --
but they now share this base rather than each carrying a private copy of the
same helpers.

Strategy contract
-----------------
``update(enX, enY, enE, **kwargs) -> np.ndarray | None``
    Return the state update step, shape ``(nx, ne)``, or ``None`` if the
    strategy declined to produce one.

Strategies read the surrounding scheme's configuration off ``self`` -- the
damping parameter ``lam``, ``trunc_energy``, ``localization``, ``keys_da``, and
optionally ``cov_data`` / ``scale_state`` / ``scale_data`` / ``proj``.
``full_update`` reads more still: ``prior_enX``, ``Am``, ``ext_Am`` and
``state_scaling``. Note that ``prior_enX`` is *ensemble* state -- it resolves
under the mixin only because the scheme delegates unknown reads to its
ensemble, so the context spans both objects.

That coupling is inherited from the mixin design and is what a later phase
replaces with an explicit context object. :meth:`AnalysisStrategy.__getattr__`
is the intermediate step: a strategy can now be *bound* to a scheme and reach
the same context by delegation, which is what allows the flavour to become a
parameter rather than part of the class name.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import solve as _dense_solve
from scipy.linalg import sqrtm as _dense_sqrtm

__all__ = ["AnalysisStrategy"]


class AnalysisStrategy(ABC):
    """Base class for analysis-step strategies.

    Provides the linear-algebra helpers every flavour needs. Both accept either
    a full 2-D matrix or a 1-D array holding just the diagonal, which is how
    PIPT represents a diagonal data covariance without materialising ``nd x nd``
    zeros.

    Two usages
    ----------
    **Mixed in** (what every shipped scheme still does)::

        class esmda_approx(esmdaMixIn, approx_update): ...

    ``self`` is the scheme, so ``self.lam`` and friends resolve by inheritance
    and nothing here is involved.

    **Bound** -- constructed against a scheme it holds a reference to::

        strategy = approx_update(scheme)
        step = strategy.update(enX, enY, enE)

    which is what lets the flavour become a *parameter* of one scheme class
    rather than picking which class you get. Context reads then fall through to
    the bound scheme via :meth:`__getattr__`, the same delegation
    :class:`~pipt.update_schemes.core.AssimilationSchemeBase` uses to
    reach its ensemble.

    An unbound strategy resolves nothing and raises ``AttributeError``, which is
    deliberate: the optional context reads below are written as
    ``getattr(self, 'scale_state', <default>)`` and must keep falling back to
    their defaults rather than finding a half-initialised scheme.
    """

    def __init__(self, scheme=None):
        """
        Parameters
        ----------
        scheme : object, optional
            Scheme to read analysis context from. ``None`` leaves the strategy
            unbound. Never invoked in the mixin case: no ``__init__`` in that
            MRO chains to ``super()``.
        """
        self._scheme = scheme

    def __getattr__(self, name):
        """Fall back to the bound scheme for context this strategy lacks.

        Only reached when normal lookup fails, so a mixed-in strategy -- where
        ``self`` is the scheme -- never gets here for an attribute that exists.
        """
        # Guard the recursion: resolving `_scheme` must not re-enter this.
        if name.startswith("__") or name == "_scheme":
            raise AttributeError(name)
        try:
            scheme = object.__getattribute__(self, "_scheme")
        except AttributeError:
            raise AttributeError(name) from None
        if scheme is None:
            raise AttributeError(name)
        return getattr(scheme, name)

    def __setattr__(self, name, value):
        """Write public attributes through to the bound scheme.

        Some strategies deliver their result by *assignment* rather than by
        return value: ``subspace_update`` sets ``w_step``, which is what the
        scheme actually applies, and ``full_update`` caches ``Am``. Mixed in,
        those writes landed on the scheme because ``self`` was the scheme. Bound,
        they would land here instead and the scheme's ``hasattr(self, 'w_step')``
        would silently be False -- the update quietly skipped, no error.

        So write-through is what makes binding faithful, not a convenience.
        Private names stay local, which is what keeps ``_scheme`` itself out of
        the loop.
        """
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        scheme = getattr(self, "_scheme", None)
        if scheme is None:
            object.__setattr__(self, name, value)
        else:
            setattr(scheme, name, value)

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
            Strategy-specific extras, e.g. ``prior`` or ``enAdj``.

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
