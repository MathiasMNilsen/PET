"""
EnRML type schemes
"""
# External imports
import pipt.misc_tools.extract_tools as extract

from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.core import AssimilationScheme, StepReport
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update
import numpy as np
import copy as cp

# `analysis/margis.py` ships a real (if unfinished -- see its module
# docstring) port of the margIS math, not an inert placeholder. The import is
# still guarded in case a private overlay replaces the module with a complete
# implementation.
#
# NOTE: this used to walk `update_methods_ns` with pkgutil so a private
# namespace package could drop a module in alongside it. That package is now
# `pipt.update_schemes.analysis`, so a private overlay must target the new
# name; the walk itself is gone, since executing every module in the package to
# discover one class is a costly way to express an optional import.
try:
    from pipt.update_schemes.analysis.margis import margIS_update
except ImportError:  # pragma: no cover - depends on a package outside this repo
    class margIS_update:
        pass


__all__ = [
    'LMEnRML',
    'GNEnRML',
]


class LMEnRML(AssimilationScheme):
    """Levenberg-Marquardt Ensemble Randomized Maximum Likelihood (LM-EnRML).

    An iterative ensemble smoother that solves the randomized maximum
    likelihood problem by repeated linearisation, with a Levenberg-Marquardt
    damping parameter :math:`\\lambda` controlling the step size. The damped
    update inflates the Hessian approximation:

    .. math::

        m \\leftarrow m + C_{md} \\big((1 + \\lambda) C_d + C_{dd}\\big)^{-1}
        (d_{obs} - g(m))

    Unlike ES-MDA, steps are accepted or rejected. A step that increases the
    mean data misfit is discarded, :math:`\\lambda` is multiplied by
    ``lambda_factor`` and the step re-solved from the same state; one that
    decreases it is kept and :math:`\\lambda` reduced. That retry loop lives
    inside :meth:`update_step`, so one iteration is one call however many
    attempts it takes -- the shape popt's optimizers have. The run stops when
    the relative misfit change falls below ``data_misfit_tol``, when
    :math:`\\lambda` reaches ``lambda_max``, when a single iteration exhausts
    ``max_inner_iter`` attempts, or on ``max_iter``.

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
    Configured through the ``iteration`` block of ``keys_da``:

    ``max_iter``
        Maximum accepted iterations.
    ``lambda``
        Initial damping parameter (default 100). ``'auto'`` derives it from the
        prior data misfit.
    ``lambda_factor``
        Factor by which damping grows on rejection and shrinks on acceptance
        (default 5). Held as ``lam_factor`` -- not ``gamma``, which is
        GN-EnRML's step length, a different quantity entirely.
    ``lambda_max``, ``lambda_min``
        Bounds on the damping parameter.
    ``max_inner_iter``
        Damping attempts one iteration may make before the run gives up
        (default 10). ``lambda_max`` normally stops it first.
    ``data_misfit_tol``
        Relative misfit change treated as converged (default 0.01).

    Examples
    --------
    >>> result = LMEnRML.assimilate(keys_da, keys_en, flow(keys_sim))
    >>> result.message
    'Maximum number of iterations reached'

    ``success`` distinguishes the two ways a run can end: ``True`` when a
    convergence criterion fired, ``False`` when ``max_iter`` was reached first.
    Both are ordinary outcomes -- check ``prior_data_misfit`` against
    ``data_misfit`` to judge whether the run achieved anything.

    References
    ----------
    Chen and Oliver, *Levenberg-Marquardt forms of the iterative ensemble
    smoother for efficient history matching and uncertainty quantification*
    [`chen2013`][].

    See Also
    --------
    GNEnRML : Gauss-Newton form, damped by a step length instead.
    ESMDA : Fixed schedule rather than convergence-driven iteration.
    """

    COMPATIBLE_ANALYSES = {
        "approx": approx_update,
        "full": full_update,
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
        # AssimilationScheme's `misfit_tol`/`step_tol` docs for why.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0)

        # Flavour is a parameter, so it selects an analysis object not a class.
        self.bind_analysis(self.resolve_analysis(analysis, keys_da))

        if self.restart is False:

            # Set parameters needed for LM-EnRML
            options = self.keys_da['iteration']
            if isinstance(options, list):
                options = extract.list_to_dict(options)

            # ------------------------------------------------------------
            # LM-EnRML Options
            # ------------------------------------------------------------
            self.data_misfit_tol = options.get('data_misfit_tol', 0.01)
            self.trunc_energy = options.get('energy', 0.95)
            self.lam       = options.get('lambda', 100)
            self.lam_max   = options.get('lambda_max', 1e10)
            self.lam_min   = options.get('lambda_min', 0.01)
            self.lam_factor = options.get('lambda_factor', 5)
            # How many times one iteration may re-damp before giving up. The
            # damping loop lives inside update_step(), so this bounds it
            # there rather than relying on the base loop's rejected-step
            # valve; `lambda_max` is normally what stops it first.
            self.max_inner_iter = options.get('max_inner_iter', 10)
            # ------------------------------------------------------------

            # Ensure that it is given as percentage
            if self.trunc_energy > 1:
                self.trunc_energy /= 100.

            # Initalize some variables
            self.iteration = 0
            # Mirrored for ensemble-side helpers that consult it.
            self.ensemble.iteration = 0
            # The prior forecast is no longer one of the counted iterations,
            # so the loop budget is one less than the legacy max_iter.
            self.max_iter = extract.extract_maxiter(self.keys_da)
            self.maxiter = self.max_iter - 1
            self._converged = False
            self.ensemble.prior_enX = cp.deepcopy(self.enX) # (Not sure if this is wise!)
            self.prev_data_misfit_mean = None  # Data misfit at previous iteration
            self.ensemble.list_datatypes = list(self.data_df.columns)

            # Load ACTNUM if given
            self.actnum = None
            if 'actnum' in self.keys_da.keys():
                try:
                    self.actnum = np.load(self.keys_da['actnum'])['actnum']
                except Exception:
                    print('ACTNUM file cannot be loaded!')

            # At the moment, the iterative loop is threated as an iterative smoother and thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.ensemble.check_assimindex_simultaneous()
            self.ensemble.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

            # Get the perturbed observations and scaling
            self.data_random_state = cp.deepcopy(np.random.get_state())
            self.vecObs = self.data_df.to_matrix()
            self.enObs = self.ensemble.perturb_observations(self.vecObs)
            self.ensemble._ext_scaling()



    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.
        """
        # Get Ensemble of predicted data
        self.enPred = self.pred_data.to_matrix()

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
            # The one path that still writes ensemble.enX_temp, which nothing
            # reads now -- so take its result explicitly.
            proposed = getattr(self.ensemble, "enX_temp", None)
            self.enX_proposal = self.enX if proposed is None else proposed
        else:

            # Check for adjoint
            if hasattr(self, 'adjoints'):
                enAdj = self.adjoints.to_matrix(is_jacobian=True) # In this case: Shape (ny, nx, ne)
            else:
                enAdj = None

            # Perform the update
            self.step = self.update(
                enX = self.enX,
                enY = self.enPred,
                enE = self.enObs,
                # kwargs
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
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX.indices}
            self.enX_proposal.clip_matrix(limits)

    # ------------------------------------------------------------------
    # AssimilationScheme contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one LM-EnRML iteration, re-damping until it finds a step.

        The damping loop is here rather than in the base loop: one call is one
        iteration, and the :math:`\\lambda` attempts it took to get there are
        this scheme's business. That mirrors popt, where ``EnOpt.update_step``
        backtracks over its own step length and returns only once it has an
        improving step or has run out of attempts.

        Each attempt re-solves the analysis at the current :math:`\\lambda`,
        forecasts the proposal and scores it. A worse misfit multiplies
        :math:`\\lambda` by ``lambda_factor`` and tries again from the *same*
        state -- nothing was committed -- so the retries cost forecasts, not
        correctness.

        Returns
        -------
        StepReport
            ``accepted`` is whether an attempt improved the misfit. It is
            ``False`` only when the scheme has also decided to stop, which
            :meth:`check_convergence` then reports to the loop.
        """
        attempt = 0
        while True:
            self.calc_analysis()
            self.after_analysis()
            state = self.run_forecast(self.enX_proposal)
            self.score_and_commit()

            if self.step_accepted or self._converged:
                break

            attempt += 1
            if attempt >= self.max_inner_iter:
                # Reported the way `lambda_max` is -- a stopping criterion
                # with its reason in `why_stop` -- because it is the same
                # event: no smaller step left to try.
                self._converged = True
                self.conv_msg = (f"No improving step after {attempt} damping "
                                 f"attempts (λ = {self.lam:.3g})")
                self.why_stop['inner_stop'] = True
                self.logger.info(self.conv_msg)
                break

        return StepReport(accepted=self.step_accepted, misfit=self.ensemble_misfit,
                          state=state)

    def score(self, pred_data=None):
        r"""Data misfit, sizing ``lambda='auto'`` the first time there is one.

        :math:`\lambda_0 = \Phi_{prior} / 2 N_d` is defined against the prior
        misfit, so it cannot be settled in ``__init__``. The first score of a
        run is the prior's, which makes this the earliest point it can be
        resolved -- and everything downstream needs a number: the prior row
        reports λ, and the prior QA/QC pass computes with it.
        """
        misfit = super().score(pred_data)
        if self.lam == 'auto' and misfit is not None:
            self.lam = 0.5 * float(np.mean(misfit)) / self.enObs.shape[0]
        return misfit

    def check_convergence(self) -> bool:
        """Report the verdict reached by the preceding :meth:`score_and_commit`."""
        return self._converged

    def score_and_commit(self):
        """
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter.

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """
        # Initialize the initial success value
        success = False

        # The λ this attempt was damped with. Captured before the branches
        # below adjust it, because that is what the row for this iteration
        # reports -- the loop logs after the adjustment has happened.
        self.lam_used = self.lam

        # if inital conv. check, there are no prev_data_misfit
        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std
        self.prev_ensemble_misfit = getattr(self, "ensemble_misfit", None)

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
        # data instead.

        data_misfit = self.score()
        self.ensemble_misfit = data_misfit
        self.data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit_mean = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit_mean / self.prev_data_misfit_mean)) < self.data_misfit_tol \
                or self.lam >= self.lam_max:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            if self.data_misfit_mean >= self.prev_data_misfit_mean:
                success = False
                self.logger(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.prev_data_misfit_mean:0.1f}'
                )
            else:
                self.logger.info(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}'
                )

            self._converged = True
            # Without this the run reports "no stopping reason recorded" on a
            # perfectly ordinary convergence: only the base class's generic
            # criteria set conv_msg, and these schemes disable those.
            self.conv_msg = (
                f"Data misfit change satisfies |1 - d/d_prev| < "
                f"{self.data_misfit_tol}"
                if abs(1 - (self.data_misfit_mean / self.prev_data_misfit_mean))
                < self.data_misfit_tol
                else f"Damping parameter reached lambda_max ({self.lam_max})"
            )
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}


            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            # If reduction in mean data misfit, reduce damping param
            if self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std < self.prev_data_misfit_std:

                success = True

                # Reduce damping parameter
                if self.lam > self.lam_min:
                    self.lam = self.lam / self.lam_factor
                    self.logger(f'λ reduced: {self.lam * self.lam_factor} ──> {self.lam}')

                # Update ensemble weights
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)


            elif self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std >= self.prev_data_misfit_std:

                # accept itaration, but keep lam the same
                success = True

                # Update ensemble weights
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                success = False
                self.lam = self.lam * self.lam_factor
                # Increase damping parameter (divide calculations for ANALYSISDEBUG purpose)
                self.logger(f'Data misfit increased! λ increased: {self.lam / self.lam_factor} ──> {self.lam}')

            if not success:
                # Back to the last accepted misfit, array included -- that is
                # what update_step reports and the next comparison uses.
                self.data_misfit_mean = self.prev_data_misfit_mean
                self.data_misfit_std = self.prev_data_misfit_std
                if self.prev_ensemble_misfit is not None:
                    self.ensemble_misfit = self.prev_ensemble_misfit

            self._converged = False
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

    def log_columns(self, prior_run: bool = False) -> dict:
        """LM-EnRML reports the damping the logged iteration ran with."""
        return {"λ": getattr(self, "lam_used", self.lam)}





class GNEnRML(AssimilationScheme):
    """Gauss-Newton Ensemble Randomized Maximum Likelihood (GN-EnRML).

    Solves the same randomized maximum likelihood problem as :class:`LMEnRML`,
    but takes undamped Gauss-Newton steps scaled by a step length
    :math:`\\gamma \\in (0, 1]` rather than inflating the Hessian:

    .. math::

        m \\leftarrow m + \\gamma \\, C_{md} (C_d + C_{dd})^{-1}
        (d_{obs} - g(m))

    Steps are accepted or rejected on the mean data misfit as in LM-EnRML. On
    acceptance :math:`\\gamma` is relaxed towards ``gamma_max``; on rejection it
    is divided by ``gamma_factor`` and the step re-solved, in the same
    within-:meth:`update_step` loop LM-EnRML uses for :math:`\\lambda`.

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
    Configured through the ``iteration`` block of ``keys_da``:

    ``max_iter``
        Maximum accepted iterations.
    ``gamma``
        Initial step length (default 0.2).
    ``gamma_max``
        Value the step length relaxes towards on success (default 0.5).
    ``gamma_factor``
        Divisor applied to the step length on rejection (default 2.5).
    ``max_inner_iter``
        Step-length attempts one iteration may make before the run gives up
        (default 10). There is no ``gamma_min``, so this is what bounds it.
    ``data_misfit_tol``
        Relative misfit change treated as converged (default 0.01).

    The ``margis`` flavour is backed by ``margIS_update``, ported from an
    older layout. It delivers its result via ``self.W_step`` (capital W) --
    the matrix-form ensemble update, distinct from the ``w_step`` most other
    flavours use -- which this method's own ``calc_analysis`` (below) handles
    with its own reconstruction branch. Run against real data it produces a
    large, sensible misfit reduction, but is still one run on one case with
    no committed reference pinning it -- see its module docstring
    (:mod:`pipt.update_schemes.analysis.margis`) for what was fixed in the
    port and what remains a modelling choice rather than a bug.

    Examples
    --------
    >>> result = GNEnRML.assimilate(keys_da, keys_en, flow(keys_sim))

    References
    ----------
    Chen and Oliver [`chen2013`][]; see also Raanes, Stordal and Evensen,
    *Revising the stochastic iterative ensemble smoother* [`raanes2019`][], and
    Evensen et al. [`evensen2019`][].

    See Also
    --------
    LMEnRML : Levenberg-Marquardt form, damped via the Hessian.
    """

    COMPATIBLE_ANALYSES = {
        "approx": approx_update,
        "full": full_update,
        "subspace": subspace_update,
        "margis": margIS_update,
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
        # AssimilationScheme's `misfit_tol`/`step_tol` docs for why.
        super().__init__(ensemble, misfit_tol=0.0, step_tol=0.0)

        # Flavour is a parameter, so it selects an analysis object not a class.
        self.bind_analysis(self.resolve_analysis(analysis, keys_da))

        if self.restart is False:
            options = self.keys_da['iteration']
            if isinstance(options, list):
                options = extract.list_to_dict(options)

            self.data_misfit_tol = options.get('data_misfit_tol', 0.01)
            self.trunc_energy = options.get('energy', 0.95)
            self.gamma = options.get('gamma', 0.2)
            self.gamma_max = options.get('gamma_max', 0.5)
            self.gamma_factor = options.get('gamma_factor', 2.5)
            # How many times one iteration may shorten the step before giving
            # up. The step-length loop lives inside update_step(), so this is
            # what bounds it; unlike LM-EnRML's `lambda_max` there is no bound
            # on gamma itself to stop it first.
            self.max_inner_iter = options.get('max_inner_iter', 10)

            # 'auto' means "pick a sensible default", which for the step
            # length is a constant -- it needs nothing from the prior, so it
            # is resolved here rather than after the prior forecast.
            if self.gamma == 'auto':
                self.gamma = 0.1

            if self.trunc_energy > 1:
                self.trunc_energy /= 100.

            self.iteration = 0
            # Mirrored for ensemble-side helpers that consult it.
            self.ensemble.iteration = 0
            # The prior forecast is no longer one of the counted iterations,
            # so the loop budget is one less than the legacy max_iter.
            self.max_iter = extract.extract_maxiter(self.keys_da)
            self.maxiter = self.max_iter - 1
            self._converged = False
            self.ensemble.prior_enX = cp.deepcopy(self.enX)
            self.prev_data_misfit_mean = None
            self.ensemble.list_datatypes = list(self.data_df.columns)

            self.actnum = None
            if 'actnum' in self.keys_da.keys():
                try:
                    self.actnum = np.load(self.keys_da['actnum'])['actnum']
                except Exception:
                    print('ACTNUM file cannot be loaded!')

            # At the moment, the iterative loop is threated as an iterative smoother and thus we check if assim. indices
            # are given as in the Simultaneous loop.
            self.ensemble.check_assimindex_simultaneous()
            self.ensemble.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

            self.data_random_state = cp.deepcopy(np.random.get_state())
            self.vecObs = self.data_df.to_matrix()
            self.enObs = self.ensemble.perturb_observations(self.vecObs)
            self.ensemble._ext_scaling()

            # ensure that the updates does not invoke the LM inflation of the Hessian.
            self.lam = 0

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.

        """

        self.enPred = self.pred_data.to_matrix()

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
            # The one path that still writes ensemble.enX_temp, which nothing
            # reads now -- so take its result explicitly.
            proposed = getattr(self.ensemble, "enX_temp", None)
            self.enX_proposal = self.enX if proposed is None else proposed
        else:

            if hasattr(self, 'adjoints'):
                enAdj = self.adjoints.to_matrix(is_jacobian=True)
            else:
                enAdj = None

            self.step = self.update(
                enX=self.enX,
                enY=self.enPred,
                enE=self.enObs,
                prior=self.prior_enX,
                enAdj=enAdj
            )

            if self.step is not None:
                self.enX_proposal = self.enX + self.gamma * self.step
            # Vector update following e.g. Evensen et al. 2019, for the
            # additive-anomaly flavours (subspace_update and friends).
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.gamma * self.w_step
                self.enX_proposal = np.dot(self.prior_enX, (np.eye(self.ne) + self.W / np.sqrt(self.ne - 1)))
            # Matrix update following e.g. Raanes et al. 2019, for flavours
            # that deliver a multiplicative ensemble-transform matrix instead
            # (margIS_update: W_0 = I, not the w_step branch's W_0 = 0).
            if hasattr(self, 'W_step'):
                self.W = self.current_W + self.gamma * self.W_step
                X_p = self.prior_enX @ self.proj * np.sqrt(self.ne - 1)
                self.enX_proposal = np.mean(self.prior_enX, axis=1, keepdims=True) + np.dot(X_p, self.W)

            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX.indices}
            self.enX_proposal.clip_matrix(limits)

    # ------------------------------------------------------------------
    # AssimilationScheme contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one GN-EnRML iteration, shortening the step until it improves.

        The same shape as :meth:`LMEnRML.update_step` -- one call is one
        iteration, and the attempts within it are this scheme's business --
        with the step length :math:`\\gamma` doing what :math:`\\lambda` does
        there. A rejected attempt divides :math:`\\gamma` by ``gamma_factor``
        and re-solves from the same state.

        Returns
        -------
        StepReport
            ``accepted`` is whether an attempt improved the misfit. It is
            ``False`` only when the scheme has also decided to stop, which
            :meth:`check_convergence` then reports to the loop.
        """
        attempt = 0
        while True:
            self.calc_analysis()
            self.after_analysis()
            state = self.run_forecast(self.enX_proposal)
            self.score_and_commit()

            if self.step_accepted or self._converged:
                break

            attempt += 1
            if attempt >= self.max_inner_iter:
                # γ has no lower bound, so this is what stops the scheme from
                # halving a step that is already far too small to matter.
                self._converged = True
                self.conv_msg = (f"No improving step after {attempt} "
                                 f"step-length attempts (γ = {self.gamma:.3g})")
                self.why_stop['inner_stop'] = True
                self.logger.info(self.conv_msg)
                break

        return StepReport(
            accepted=self.step_accepted,
            misfit=self.ensemble_misfit,
            state=state
        )

    def check_convergence(self) -> bool:
        """Report the verdict reached by the preceding :meth:`score_and_commit`."""
        return self._converged

    def score_and_commit(self):
        """
        Check if LM-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter.

        Returns
        -------
        conv: bool
            Logic variable telling if algorithm has converged
        why_stop: dict
            Dict. with keys corresponding to conv. criteria, with logical variable telling which of them that has been
            met
        """
        # Initialize the initial success value
        success = False

        # The γ this attempt took, captured before the branches below relax
        # or shorten it -- see LMEnRML.score_and_commit.
        self.gamma_used = self.gamma

        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std
        self.prev_ensemble_misfit = getattr(self, "ensemble_misfit", None)

        data_misfit = self.score()
        self.ensemble_misfit = data_misfit

        self.data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit_mean = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit_mean / self.prev_data_misfit_mean)) < self.data_misfit_tol:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'gamma': self.gamma,
                        }

            if self.data_misfit_mean >= self.prev_data_misfit_mean:
                success = False
                self.logger.info(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.prev_data_misfit_mean:0.1f}')
            else:
                self.logger.info(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}')
            self._converged = True
            # Without this the run reports "no stopping reason recorded" on a
            # perfectly ordinary convergence: only the base class's generic
            # criteria set conv_msg, and these schemes disable those.
            self.conv_msg = (
                f"Data misfit change satisfies |1 - d/d_prev| < "
                f"{self.data_misfit_tol}"
            )
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'gamma': self.gamma}

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            # If reduction in mean data misfit, reduce damping param
            if self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std < self.prev_data_misfit_std:
                success = True

                if self.gamma_factor > 1:
                    self.gamma = self.gamma + (self.gamma_max - self.gamma) * 2 ** (
                        -(self.iteration + 1) / (self.gamma_factor - 1)
                    )

                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            elif self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std >= self.prev_data_misfit_std:
                # accept itaration, but keep lam the same
                success = True

                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                success = False

                if self.gamma_factor > 1:
                    self.gamma = self.gamma / self.gamma_factor

                self.logger(
                    f'Data misfit increased! New Gamma for repeated analysis: {self.gamma}'
                )

            if not success:
                # Back to the last accepted misfit, array included.
                self.data_misfit_mean = self.prev_data_misfit_mean
                self.data_misfit_std = self.prev_data_misfit_std
                if self.prev_ensemble_misfit is not None:
                    self.ensemble_misfit = self.prev_ensemble_misfit

            self._converged = False
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

    def log_columns(self, prior_run: bool = False) -> dict:
        """GN-EnRML reports the step length the logged iteration took."""
        return {"γ": getattr(self, "gamma_used", self.gamma)}


class co_lm_enrml(LMEnRML):
    """Approximate LM-EnRML of Chen and Oliver (2013), under its historical name.

    This is ``LMEnRML(..., analysis="approx")`` and nothing more: the class
    only ever differed from LM-EnRML by mixing in the approximate analysis,
    which is a constructor argument now. It stays so that configs written as
    ``scheme = "co_lm_enrml"`` and code importing the name keep working.
    New code should say ``LMEnRML`` with ``analysis="approx"``.
    """

    COMPATIBLE_ANALYSES = {"approx": approx_update}

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        # The name pins the flavour, so a config that does not say gets it.
        if analysis is None and "analysis" not in keys_da:
            analysis = "approx"
        super().__init__(keys_da, keys_en, sim, analysis=analysis)


class gn_enrml(GNEnRML):
    """Gauss-Newton stochastic IES of Raanes et al. (2019), under its historical name.

    This is ``GNEnRML(..., analysis="subspace")``: the weight-space update this
    class used to carry inline is the ``subspace`` analysis, and the
    step-length schedule it called ``lambda`` is GN-EnRML's ``gamma``
    schedule. It stays so that configs written as ``scheme = "gn_enrml"`` and
    code importing the name keep working. New code should say ``GNEnRML``
    with ``analysis="subspace"``.
    """

    COMPATIBLE_ANALYSES = {"subspace": subspace_update}

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        if analysis is None and "analysis" not in keys_da:
            analysis = "subspace"
        super().__init__(keys_da, keys_en, sim, analysis=analysis)
