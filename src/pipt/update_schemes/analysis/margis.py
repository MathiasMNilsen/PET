"""Stochastic iterative ensemble smoother (IES, i.e. EnRML) with *subspace* implementation.

Ported from ``update_methods_ns/margIS_update.py`` on the project's ``main``
branch (an older, pre-refactor layout), replacing the inert placeholder that
used to live here. This is closer to real than that placeholder -- it reads
attribute names (``self.ne``, ``self.proj``, ``self.lam``, ``self.scale_data``)
that match this codebase's current conventions, and its
``update(self, enX, enY, enE, **kwargs)`` signature matches what
``GNEnRML.calc_analysis`` already calls it with -- unlike on ``main``, where
the equivalent caller passes no arguments at all.

Several problems in the ported code have been fixed here, against
Stordal, Lorentzen & Fossum, *Marginalized iterative ensemble smoothers for
data assimilation*, Computational Geosciences 27:975-986 (2023). One of
these was diagnosed wrong on the first pass and is recorded here so the
mistake is not repeated:

- It delivers its result via ``self.W_step`` (capital W), the ensemble
  *matrix* update ("following e.g. Raanes et al. 2019", per the code this
  was ported from), whose reconstruction is
  ``enX = mean(prior_enX) + prior_enX @ proj * sqrt(ne-1) @ W``. That branch
  had been dropped from this codebase's ``GNEnRML.calc_analysis`` -- only
  the lowercase ``w_step`` *vector* update ("following e.g. Evensen et al.
  2019", reconstruction ``enX = prior_enX @ (I + W/sqrt(ne-1))``) remained.
  The first fix here renamed ``self.W_step`` to ``self.w_step`` to match the
  branch that still existed -- which was wrong: it is a different formula
  for a differently-defined ``W`` (this method's ``W`` starts at the identity
  per the paper, Section 2.4; the vector update's starts at zero), not an
  alternative name for the same one. Confirmed by running it: routed through
  the vector-update branch, the assimilation made the misfit *worse* by five
  orders of magnitude, and stayed exactly as bad regardless of how small the
  step length ``gamma`` shrank -- the signature of applying the wrong
  reconstruction formula entirely, not a scale problem. The real fix restores
  the missing ``hasattr(self, 'W_step')`` branch to ``GNEnRML.calc_analysis``
  (see there) and leaves this file delivering ``self.W_step`` as it always
  did. Confirmed against real data (PIPT's own ``TinyBox`` tutorial case):
  misfit prior 1.96e10, after one iteration 1.18e8, a 99.4% reduction.
- The update loop was hardcoded to 70 individual data points, each its own
  "type" of one (``M = 1``), matching neither the data actually being
  assimilated nor the method's own general form. Equations 8-9 of the paper
  give the multi-type log-likelihood as a *sum over data types*, each with
  its own count ``M_k`` -- Eq. 37's ``(M + nu)/(S + nu*s**2)`` factor (what
  ``Ratio`` computes below) is exactly one term of that sum. The loop now
  groups rows by data type (``self.data_df``'s columns) instead of walking
  points one at a time; ``M`` is each type's actual row count rather than a
  fixed ``1``.
- It checked ``if self.iteration == 1`` to detect the first call and
  initialise ``current_W``/``current_w``/``D``. This codebase's schemes count
  from ``self.iteration = 0`` (the log even prints ``self.iteration + 1`` to
  display 1-based numbers), so the first real analysis call happens at
  ``iteration == 0`` -- confirmed against ``GNEnRML.__init__`` and
  ``subspace_update``, which does the same ``if self.iteration == 0`` check
  for the same reason. Left at ``== 1`` (the ported code's convention, from a
  layout that apparently counted from 1), initialisation never ran and the
  first real call failed outright with ``AttributeError: 'AssimilationEnsemble'
  object has no attribute 'current_W'``.
- It carried its own ``scale()`` (elementwise for a diagonal covariance,
  else a dense solve), duplicating :meth:`AnalysisStrategy.solve` -- the same
  duplication ``approx``/``full``/``subspace`` used to have before they were
  consolidated onto the shared base (see that base's module docstring). Now
  ``margIS_update`` inherits :class:`AnalysisStrategy` and calls ``self.solve``
  directly, picking up the same fix that consolidation made: ``np.ndim``
  rather than ``scaling.shape``, so a covariance passed as a plain list or
  scalar works rather than raising ``AttributeError``.

``nu``/``s`` remain a single shared value across all types rather than
per-type ``nu_k``/``s_k`` -- the paper's own worked example (Section 3) does
the same, setting one shared ``nu`` (there, the total measurement count) for
every type, so this is not a shortcut introduced here.

Inheriting ``AnalysisStrategy`` also let ``"margis": margIS_update`` join
``GNEnRML.COMPATIBLE_ANALYSES`` directly, the same way ``"approx"`` and
friends are listed there -- ``GNEnRML(..., analysis="margis")`` builds
``margIS_update(self)`` by ordinary composition, no mixin involved. The
former ``gnenrml_margis`` class -- which mixed ``margIS_update`` into its
bases instead -- is gone; while it existed, that mixing turned out to be
broken in its own right (before this class-level entry existed): with
``GNEnRML`` listed first, plain attribute lookup found ``StrategyMixin.update``
before ``margIS_update.update``, so the scheme could not run regardless of
this file's own math. See :class:`pipt.update_schemes.core.strategy.StrategyMixin`
for why that shadowing happens and why binding avoids it entirely.

This has now been run against real data (see above) and produces a large,
sensible misfit reduction on one case -- worth far more confidence than "it
runs without erroring," but still not a golden reference: it is one run, on
one case, with no committed values pinning today's numbers against a future
change the way :mod:`test_numerical_characterisation` does for the other
flavours. Treat it as plausible, not verified.
"""

import numpy as np
import pandas as pd

import pipt.misc_tools.analysis_tools as at
from pipt.update_schemes.analysis.base import AnalysisStrategy


def _row_datatypes(df):
    """Datatype label for each row ``df.to_matrix()`` produces, in that order.

    ``PETDataFrame.to_matrix()`` flattens time-major, interleaving data types
    within each time step, and drops any all-missing (time, datatype) cell --
    so datatype rows are neither contiguous nor evenly spaced, and cannot be
    recovered by striding. This mirrors ``to_matrix()``'s own filtering and
    per-cell array expansion exactly, over the ``(index, datatype)`` labels
    ``to_series()`` already carries, so the result lines up one-to-one with
    ``to_matrix()``'s rows.
    """
    labels = []
    for (_, datatype), val in df.to_series().items():
        if not np.any(pd.notna(np.atleast_1d(val))):
            continue
        if (not df.is_ensemble) and isinstance(val, np.ndarray):
            labels.extend([datatype] * len(val))
        else:
            labels.append(datatype)
    return labels


class margIS_update(AnalysisStrategy):
    """
    MargIES update from Stordal et.al.
    This is now implemented with perturbed observations, which means that we set a prior belief on the data uncertainty.
    Thus, the prior is an invers chi2 distriubtuinm and after scaling the mean varians is 1.
    """

    def update(self, enX, enY, enE, **kwargs):

        if self.iteration == 0:  # method requires some initiallization
            self.current_W = np.eye(self.ne)
            self.current_w = np.zeros(self.ne)
            self.D = self.solve(self.scale_data, enE)
            # Scale everything so that data uncertainty is I

        sY = self.solve(self.scale_data, enY) #Scaling is same as with 'known' uncertainty, hence makes sense to set s = 1
        self.S = 0

        deltaD = 0
        deltaD_sqrt = 0

        Y = np.linalg.solve(self.current_W.T, sY.T).T
        Y = Y @ self.proj * np.sqrt(self.ne - 1)

        # One term of Eq. 8/9 per data type, not per individual point.
        row_labels = np.asarray(_row_datatypes(self.data_df))
        data_types = pd.unique(row_labels)
        s = 1 #should be default option with possibility to change in setup
        nu = self.ne-1 #should be default option with possibility to change in setup
        for dtype in data_types:
            index = np.flatnonzero(row_labels == dtype)
            M = len(index)  # Numbers of data of this type.

            delta = self.D[index,:]-sY[index,:]
            Chi = np.sum(delta * delta, axis = 0)
            Chi = np.mean(Chi)
            Ratio = (M + nu) / (Chi + nu*s*s)
            #Ratio = 1
            #Gradient
            deltaD = deltaD + (Y[index,:] * Ratio).T @ delta
            deltaD_sqrt = deltaD_sqrt + np.mean((Y[index, :] * Ratio).T @ delta ,axis=1)
            # Hessian
            self.S = self.S + (Y[index,:] * Ratio).T @ Y[index,:]

        deltaM = (self.ne-1)*(np.eye(self.ne)-self.current_W)
        deltaM_sqrt = (self.ne-1)*self.current_w
        self.S = self.S + np.eye(self.ne) * (self.ne - 1)
        Delta = deltaM + deltaD
        Delta_sqrt = deltaM_sqrt + deltaD_sqrt


        self.W_step =   np.linalg.solve(self.S, Delta) / (1 + self.lam)
       # self.sqrt_w_step = np.linalg.solve(self.S, Delta_sqrt) / (1 + self.lam)
