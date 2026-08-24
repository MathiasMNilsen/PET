"""
EnKF type schemes
"""
# External imports
import numpy as np
from copy import deepcopy
from geostat.decomp import Cholesky                     # Making realizations

# Internal imports
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.core.workflow import AssimilationScheme
from pipt.update_schemes.core.scheme_base import StepReport
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.subspace import subspace_update
# Misc. tools used in analysis schemes
from pipt.misc_tools import analysis_tools as at
import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.extract_tools as extract



class EnKF(AssimilationScheme):
    """Ensemble Kalman Filter (EnKF).

    Assimilates data sequentially, updating the state once per group of
    observations in the order given by ``assimindex``. Each update applies the
    Kalman equations with the covariances approximated from the ensemble:

    .. math::

        m \\leftarrow m + C_{md} (C_{dd} + C_d)^{-1} (d_{obs} - g(m))

    There is no damping and no rejection: every step is accepted, and the run
    ends once the data groups are exhausted.

    Parameters
    ----------
    keys_da : dict
        Parsed ``dataassim`` configuration. Besides the keys every scheme
        reads -- ``data``, ``datavar``, ``obsname``, ``truedataindex`` -- the
        ones this scheme acts on are listed under Notes.
    keys_en : dict
        Parsed ``ensemble`` configuration: ensemble size ``ne``, the ``state``
        variable names, and the ``prior_<name>`` blocks describing each.
    sim : object
        Forward simulator instance, e.g. ``simulator.opm.flow``.
    analysis : {'approx', 'full', 'subspace'}, optional
        Analysis flavour, i.e. how the ensemble-approximated sensitivity is
        inverted. Defaults to the ``analysis`` key in ``keys_da``, falling back
        to ``'approx'``. The flavours differ in cost and in how they handle a
        rank-deficient ensemble; they solve the same update equation.

    Attributes
    ----------
    ensemble : pipt.ensembles.AssimilationEnsemble
        Collaborator holding the state realisations, observed data and
        simulator. Its state is exposed as properties on the scheme, so
        ``scheme.enX`` and ``scheme.keys_da`` read straight through.
    analysis : pipt.update_schemes.analysis.AnalysisBase
        The bound analysis object. Note the constructor takes ``analysis`` as
        a *name* and this attribute holds the resulting object, the way
        ``Model(optimizer="adam").optimizer`` is an optimizer instance.
    analysis_name : str
        The flavour name that was resolved, e.g. ``'approx'``.
    iteration : int
        Accepted iterations completed so far.
    data_misfit, prior_data_misfit : float
        Current and initial mean data misfit.

    Notes
    -----
    ``assimindex`` determines the grouping and ordering of the sequential
    updates. If all data are to be assimilated in a single step, use :class:`ES`,
    which is this scheme specialised to one group.

    ``energy`` sets the fraction of singular values retained in the truncated
    SVD (default 0.98); values above 1 are read as percentages.

    Every data group is assimilated exactly once, so the prior-increment term
    that distinguishes ``full`` from ``approx`` is never reached: ``"full"``
    is pointed at the same class as ``"approx"`` in
    :attr:`COMPATIBLE_ANALYSES`. :class:`ES` inherits this.

    Examples
    --------
    >>> result = EnKF.assimilate(keys_da, keys_en, flow(keys_sim))

    References
    ----------
    Evensen, *Data Assimilation: The Ensemble Kalman Filter* [`evensen2009a`][].

    See Also
    --------
    ES : All-data-at-once form of the same update.
    """

    # Neither this class nor ES revisit a data group, so the prior-increment
    # term "full" adds over "approx" never applies -- the two produce
    # identical output (pinned by the characterisation suite), just through
    # more expensive machinery for "full". Rather than special-case that in
    # code, "full" is simply pointed at the same class as "approx" here.
    COMPATIBLE_ANALYSES = {
        "approx": approx_update,
        "full": approx_update,
        "subspace": subspace_update,
    }

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        """Build the ensemble from the config and bind the analysis.

        See the class docstring for the parameters.
        """
        # Build the collaborator, then hand it to the scheme base -- which
        # adopts the ensemble's own logger, so log output is unchanged.
        ensemble = Ensemble(keys_da, keys_en, sim)
        # Zero tolerances switch off the base class's generic convergence
        # criteria; this scheme decides in check_convergence(). See
        # AssimilationSchemeBase's `misfit_tol`/`step_tol` docs for why.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0)

        # Flavour is a parameter, so it selects an analysis object not a class.
        self.bind_analysis(self.resolve_analysis(analysis, keys_da))

        self.prev_data_misfit_mean = None

        if self.restart is False:
            self.ensemble.prior_enX = deepcopy(self.enX)
            self.ensemble.list_states = list(self.idX.keys())

            # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.ensemble.check_assimindex_simultaneous()

            self.ensemble.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            self.ensemble.list_datatypes = self.keys_da['datatype']


            # Extract no. assimilation steps from MDA keyword in DATAASSIM part of init. file and set this equal to
            # the number of iterations pluss one. Need one additional because the iter=0 is the prior run.
            self.max_iter = len(self.keys_da['assimindex'])+1
            # Prior forecast is not a counted iteration under the base loop.
            self.maxiter = self.max_iter - 1
            self.iteration = 0
            # Mirrored for ensemble-side helpers that consult it.
            self.ensemble.iteration = 0
            self.lam = 0  # set LM lamda to zero as we are doing one full update.

            if 'energy' in self.keys_da:
                # initial energy (Remember to extract this)
                self.trunc_energy = self.keys_da['energy']
                if self.trunc_energy > 1:  # ensure that it is given as percentage
                    self.trunc_energy /= 100.
            else:
                self.trunc_energy = 0.98

            # Get the perturbed observations and observation scaling
            self.vecObs = self.data_df.to_matrix()
            self.enObs = self.ensemble.perturb_observations(self.vecObs)
            self.enObs_conv = deepcopy(self.enObs)
            self.ensemble._ext_scaling()

    def score_prior(self):
        """Score the prior forecast.

        Was an ``if self.prior_data_misfit_mean is None`` branch at the top of
        :meth:`calc_analysis`, which ran after the iteration-0 artifacts had
        already been written. ``ensemble_misfit`` is recorded here as well, so
        the per-realisation misfits are available to ``savedata`` for the
        prior as they are for every later iteration.
        """
        enPred = self.pred_data.to_matrix()

        data_misfit = at.calc_objectivefun(self.enObs, enPred, self.scale_data)

        self.ensemble_misfit = data_misfit
        self.data_misfit_mean = np.mean(data_misfit)
        self.prior_data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        self.logger.info(
            f'Prior run complete with data misfit: {self.prior_data_misfit_mean:0.1f}.')

    def calc_analysis(self):
        """
        Calculate the analysis step of the EnKF procedure. The updating is done using the Kalman filter equations, using
        svd for numerical stability. Localization is available.
        """
        # Augment observed and predicted data
        if extract.is_enabled(self.keys_da.get('emp_cov', False)):
            self.enPred = self.pred_data.to_matrix()
        else:
            self.enPred = self.pred_data.to_matrix()

            #self.cov_data = at.gen_covdata(
            #    self.datavar,
            #    self.assim_index,
            #    self.list_datatypes
           # )
            self.cov_data = at.construct_data_cov(self.data_var_df)

            generator = Cholesky()  # Initialize GeoStat class for generating realizations
            self.data_random_state = deepcopy(np.random.get_state())
            self.enObs, self.scale_data = generator.gen_real(
                self.vecObs,
                self.cov_data,
                self.ne,
                return_chol=True
            )

        self.E = np.dot(self.enObs, self.proj)

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
            # Local analysis is the one path that still writes the ensemble's
            # own enX_temp; nothing reads that field any more, so take the
            # result explicitly. (That path is flagged unimplemented since the
            # refactor -- see approx_update -- hence the fallback.)
            proposed = getattr(self.ensemble, "enX_temp", None)
            self.enX_proposal = self.enX if proposed is None else proposed
        else:
            # Check for adjoint
            if hasattr(self, 'adjoints'):
                enAdj = self.adjoints.to_matrix(is_jacobian=True) # In this case: Shape (ny, nx, ne)
            else:
                enAdj = None

            self.step = self.update(
                enX = self.enX,
                enY = self.enPred,
                enE = self.enObs,
                prior = self.prior_enX,
                enAdj = enAdj
            )
            # Update the state ensemble and weights
            if self.step is not None:
                self.enX_proposal = self.enX + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                self.enX_proposal = np.dot(self.prior_enX, (np.eye(self.ne) + self.W/np.sqrt(self.ne - 1)))

            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX.keys()}
            self.enX_proposal = entools.clip_matrix(self.enX_proposal, limits, self.idX)

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one EnKF step: analysis, forecast, then score and commit.

        Returns
        -------
        bool
            Always ``True``. The EnKF applies one update per data group and
            has no rejection path.
        """
        self.calc_analysis()
        self.after_analysis()
        state = self.run_forecast(self.enX_proposal)
        self.score_and_commit()
        return StepReport(accepted=True, misfit=self.ensemble_misfit,
                          state=state)

    def check_convergence(self) -> bool:
        """The EnKF runs its full sweep of data groups; nothing stops early."""
        return False

    def score_and_commit(self):
        """
        Calculate the "convergence" of the method. Important to
        """
        self.prev_data_misfit_mean = self.prior_data_misfit_mean

        # only calulate for the final (posterior) estimate
        if self.iteration + 1 == len(self.keys_da['assimindex']):
            enPred = self.pred_data.to_matrix()
            data_misfit = at.calc_objectivefun(self.enObs, enPred, self.scale_data)
            self.ensemble_misfit = data_misfit
            self.data_misfit_mean = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

        else:  # sequential updates not finished. Misfit is not relevant
            self.data_misfit_mean = self.prior_data_misfit_mean

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean),
                    'data_misfit': self.data_misfit_mean,
                    'prev_data_misfit': self.prev_data_misfit_mean}

        # Update state ensemble

        if self.data_misfit_mean == self.prev_data_misfit_mean:
            self.logger.info(
                f'EnKF update {self.iteration} complete!')
        else:
            if self.data_misfit_mean < self.prior_data_misfit_mean:
                self.logger.info(
                    f'EnKF update complete! Objective function decreased from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.')
            else:
                self.logger.info(
                    f'EnKF update complete! Objective function increased from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}.')
        self.why_stop = why_stop
        return why_stop


#: Historical name, kept for subclasses outside this module.
enkfMixIn = EnKF
