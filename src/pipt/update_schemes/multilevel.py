'''
Multilevel schemes developed in the 4DSeis project.

The multilevel machinery is *ensemble* work: it reorganises the state into one
block per fidelity level and configures the simulator to run them. It therefore
lives on :class:`MultilevelEnsemble`, which the scheme composes, rather than
being inherited by the scheme itself.

That split matters. ``multilevel`` previously subclassed the ensemble and
``esmda_hybrid`` inherited from both it and the ES-MDA scheme, relying on C3
linearisation to route ``super().__init__()`` into the scheme's constructor.
Once the schemes stopped inheriting the ensemble, the ensemble intercepted that
chain and the scheme's ``__init__`` silently stopped running -- leaving
``alpha`` unset and the analysis step broken. Composition removes the ordering
dependence entirely.
'''

#──────────────────────────────────────────────────────────────────────────────────────
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.esmda import ESMDA
from pipt.misc_tools import analysis_tools as at
from geostat.decomp import Cholesky
from pipt.update_schemes.analysis.hybrid import hybrid_update

import numpy as np
from copy import deepcopy
#──────────────────────────────────────────────────────────────────────────────────────


__all__ = ['MultilevelEnsemble', 'multilevel', 'esmda_hybrid']


class MultilevelEnsemble(Ensemble):
    """Ensemble whose state is partitioned into fidelity levels.

    ``enX`` is a *list* of matrices, one per level, rather than a single
    ``(nx, ne)`` matrix, and the simulator is configured to run each level.
    Everything else is the ordinary assimilation ensemble.

    Attributes
    ----------
    enX : list of ndarray
        State ensemble per level; ``enX[l]`` has shape ``(nx, ml_ne[l])``.
    tot_level : int
        Number of fidelity levels.
    ml_ne : list of int
        Ensemble size at each level.
    """

    def __init__(self, keys_da, keys_en, sim):
        super().__init__(keys_da, keys_en, sim)

        self.list_states = list(self.idX.keys())

        # Keep the unpartitioned prior: state scaling is defined over the whole
        # state, not per level. Under the previous class layout the scheme's
        # __init__ ran before the split and so saw the matrix; holding it here
        # reproduces that without depending on constructor ordering.
        self._flat_prior_enX = deepcopy(self.enX)

        # Reorganize prior ensemble to multilevel structure if nested is true
        self.enX = self.reorganize_ml_prior(self.enX)
        self.prior_enX = deepcopy(self.enX)

        # Set ML specific options for simulator
        self._init_sim()

        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
        self.list_datatypes = self.keys_da['datatype']

        self.cov_data = at.construct_data_cov(self.data_var_df)
        self.vecObs = self.data_df.to_matrix()

    def _ext_scaling(self):
        """Compute state scaling from the unpartitioned prior.

        The base implementation reads ``prior_enX.indices``, which does not
        exist once the prior is a list of per-level blocks.
        """
        self.state_scaling = at.calc_scaling(
            self._flat_prior_enX, self._flat_prior_enX.indices, self.prior_info
        )
        self.Am = None

    def _init_sim(self):
        """
        Ensure that the simulator is initiallized to handle ML forward simulation.
        """
        self.sim.multilevel = [l for l in range(self.tot_level)]
        self.sim.rawmap = [None] * self.tot_level
        self.sim.ecl_coarse = [None] * self.tot_level
        self.sim.well_cells = [None] * self.tot_level

    def reorganize_ml_prior(self, enX: np.ndarray) -> list:
        '''
        Reorganize prior ensemble to multilevel structure (list of matrices).
        '''
        ml_enX = []
        start  = 0
        for l in self.multilevel['levels']:
            stop = start + self.multilevel['ml_ne'][l]
            ml_enX.append(enX[:, start:stop])
            start = stop
        return ml_enX


#: Historical name for the multilevel container, which used to be what schemes
#: inherited. It is the ensemble now, so this is an alias rather than a base.
multilevel = MultilevelEnsemble


class esmda_hybrid(hybrid_update, ESMDA):
    '''
    A multilevel implementation of the ES-MDA algorithm with the hybrid gain.

    Composes a :class:`MultilevelEnsemble` and mixes in ``hybrid_update``, which
    supplies ``update()`` for the per-level gain. ``hybrid`` is not a registered
    analysis flavour, so no strategy is bound and the mixed-in implementation is
    used -- see :class:`pipt.update_schemes.core.StrategyMixin`.

    Notes
    -----
    Requires a ``multilevel`` block in ``keys_en`` giving ``levels``,
    ``en_size`` per level and ``ml_weights``.
    '''

    ENSEMBLE_CLASS = MultilevelEnsemble

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        super().__init__(keys_da, keys_en, sim, analysis=analysis)

        self.proj = []
        for l in range(self.tot_level):
            nl = self.ml_ne[l]
            proj_l = (np.eye(nl) - np.ones((nl, nl))/nl) / np.sqrt(nl - 1)
            self.proj.append(proj_l)

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> bool:
        """Run one multilevel ES-MDA step.

        Returns
        -------
        bool
            Always ``True``; ES-MDA takes a fixed schedule and never rejects.
        """
        self.calc_analysis()
        self.after_analysis()
        self.run_forecast()
        self.score_and_commit()
        return True

    def check_convergence(self) -> bool:
        """ES-MDA runs its full schedule of inflated steps; nothing stops early."""
        return False

    def score_prior(self):
        """Score the prior forecast across all fidelity levels.

        Same move as :meth:`pipt.update_schemes.esmda.ESMDA.score_prior`: out
        of the ``iteration == 0`` branch of :meth:`calc_analysis` and into a
        hook that runs before the iteration-0 artifacts are written.
        """
        self.enPred = [self.pred_data[l].to_matrix() for l in range(self.tot_level)]

        # Note, evaluate for high fidelity model
        data_misfit = at.calc_objectivefun(
            self.enObs_conv,
            np.concatenate(self.enPred, axis=1),  # Is this correct, given the comment above??????
            self.cov_data
        )

        self.ensemble_misfit = data_misfit
        self.prior_data_misfit = np.mean(data_misfit)
        self.prior_data_misfit_std = np.std(data_misfit)
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        self.log_update(prior_run=True)

    def calc_analysis(self):

        # Get ensemble predictions at all levels
        self.enPred = []
        for l in range(self.tot_level):
            enPred_level = self.pred_data[l].to_matrix()
            self.enPred.append(enPred_level)

        # Initialize GeoStat class for generating realizations
        cholesky = Cholesky()

        if self.iteration == 0:  # first iteration

            self.data_random_state = deepcopy(np.random.get_state())

            self.ml_enObs = []
            self.scale_data = []
            self.E = []
            for l in range(self.tot_level):

                # Generate real data and scale data
                enObs_level, scale_data_level = cholesky.gen_real(
                    self.vecObs,
                    self.alpha[self.iteration] * self.cov_data,
                    self.ml_ne[l],
                    return_chol=True
                )
                self.ml_enObs.append(enObs_level)
                self.scale_data.append(scale_data_level)
                self.E.append(np.dot(enObs_level, self.proj[l]))

        else:
            self.data_random_state = deepcopy(np.random.get_state())

            for l in range(self.tot_level):
                self.ml_enObs[l], self.scale_data[l] = cholesky.gen_real(
                    self.vecObs,
                    self.alpha[self.iteration] * self.cov_data,
                    self.ml_ne[l],
                    return_chol=True
                )
                self.E[l] = np.dot(self.ml_enObs[l], self.proj[l])

        # Calculate update step. `hybrid_update` delivers its result by
        # assigning `self.step` and returns nothing, so assigning the return
        # value here would overwrite the step it just computed with None --
        # which silently discarded every update.
        self.step = None
        returned = self.update(
            enX = self.enX,
            enY = self.enPred,
            enE = self.ml_enObs
        )
        if returned is not None:
            self.step = returned
        if self.step is not None:
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX[0].indices}
            # Written through the ensemble: the forecast reads enX_temp off the
            # collaborator, and attribute delegation covers reads only.
            enX_temp = []
            for l in range(self.tot_level):
                level = self.enX[l] + self.step[l]
                level.clip_matrix(limits)
                enX_temp.append(level)
            self.ensemble.enX_temp = enX_temp

    def score_and_commit(self):
        """Score the forecast that followed the analysis, then commit the step.

        Was the second half of ``check_convergence``. ES-MDA never tested for
        convergence there; it recomputed the misfit, logged the iteration and
        promoted ``enX_temp``.

        Returns
        -------
        dict
            The ``why_stop`` record, also stored on ``self.why_stop``.
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        enPred = []
        for l in range(self.tot_level):
            enPred_level = self.pred_data[l].to_matrix()
            enPred.append(enPred_level)

        data_misfit = at.calc_objectivefun(
            self.enObs_conv,
            np.concatenate(enPred,axis=1),
            self.cov_data
        )
        self.ensemble_misfit = data_misfit
        self.data_misfit = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        # Log update results
        success = self.data_misfit < self.prev_data_misfit
        self.log_update(success=success)

        self.ensemble.enX = deepcopy(self.enX_temp)
        self.ensemble.enX_temp = None

        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        self.why_stop = why_stop
        return why_stop
