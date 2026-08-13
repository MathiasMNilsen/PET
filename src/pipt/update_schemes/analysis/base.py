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
optionally ``cov_data`` / ``scale_state`` / ``scale_data`` / ``proj``. That
coupling is inherited from the mixin design and is what a later phase replaces
with an explicit context object.
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
    """

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
