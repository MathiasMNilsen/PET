"""
EnRML type schemes
"""
# External imports
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

from geostat.decomp import Cholesky
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.core.workflow import AssimilationScheme
from pipt.update_schemes.core.scheme_base import StepReport
from pipt.update_schemes.analysis.approx import approx_update
from pipt.update_schemes.analysis.full import full_update
from pipt.update_schemes.analysis.subspace import subspace_update
import numpy as np
import copy as cp
from scipy.linalg import cholesky, solve, inv, lu_solve, lu_factor

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

    Unlike ES-MDA, iterations are accepted or rejected. A step that increases
    the mean data misfit is discarded, :math:`\\lambda` is multiplied by
    ``lambda_factor`` and the iteration is retried; a step that decreases it is
    kept and :math:`\\lambda` reduced. The run stops when the relative misfit
    change falls below ``data_misfit_tol``, when :math:`\\lambda` reaches
    ``lambda_max``, or on ``max_iter``.

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
        (default 5).
    ``lambda_max``, ``lambda_min``
        Bounds on the damping parameter.
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
        # AssimilationSchemeBase's `misfit_tol`/`step_tol` docs for why.
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
            self.gamma     = options.get('lambda_factor', 5)
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



    def score_prior(self):
        """Score the prior forecast and size the initial damping parameter.

        Runs once, before the loop, so the iteration-0 artifacts record the
        prior misfit. Doing it here rather than behind an ``iteration == 0``
        branch in :meth:`calc_analysis` also stops a rejected first step from
        overwriting ``prior_data_misfit`` with the rejected forecast's misfit
        on every retry.
        """
        self.enPred = self.pred_data.to_matrix()

        data_misfit = at.calc_objectivefun(self.enObs, self.enPred, self.cov_data)

        self.ensemble_misfit = data_misfit
        self.data_misfit_mean = np.mean(data_misfit)
        self.prior_data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        if self.lam == 'auto':
            self.lam = (0.5 * self.prior_data_misfit_mean)/self.enPred.shape[0]

        self.log_update(success=True, prior_run=True)

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.
        """
        # Get Ensemble of predicted data
        self.enPred = self.pred_data.to_matrix()

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
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
                self.ensemble.enX_temp = self.enX + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                self.ensemble.enX_temp = np.dot(self.prior_enX, (np.eye(self.ne) + self.W/np.sqrt(self.ne - 1)))


            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX.indices}
            self.ensemble.enX_temp.clip_matrix(limits)

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one LM-EnRML step: analysis, forecast, then score and commit.

        Returns
        -------
        bool
            Whether the step was accepted. A rejected step leaves ``enX``
            untouched and backs off, so the loop retries at the same iteration
            number rather than advancing.
        """
        self.calc_analysis()
        self.after_analysis()
        self.run_forecast()
        self.score_and_commit()
        return StepReport(accepted=self.step_accepted, misfit=self.ensemble_misfit)

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
        # Get Ensemble of predicted data
        enPred = self.pred_data.to_matrix()

        # Initialize the initial success value
        success = False

        # if inital conv. check, there are no prev_data_misfit
        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std
        self.prev_ensemble_misfit = getattr(self, "ensemble_misfit", None)

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector),1)), np.ones((1, self.ne))) # use the perturbed
        # data instead.

        data_misfit = at.calc_objectivefun(self.enObs, enPred, self.cov_data)
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
                self.log_update(success=success)
                self.logger(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.prev_data_misfit_mean:0.1f}'
            )
            else:
                self.log_update(success=True)
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
                self.log_update(success=success)

                # Reduce damping parameter
                if self.lam > self.lam_min:
                    self.lam = self.lam / self.gamma
                    self.logger(f'λ reduced: {self.lam * self.gamma} ──> {self.lam}')

                # Update state ensemble
                self.ensemble.enX = cp.deepcopy(self.enX_temp)
                self.ensemble.enX_temp = None

                # Update ensemble weights
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)


            elif self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std >= self.prev_data_misfit_std:

                # accept itaration, but keep lam the same
                success = True
                self.log_update(success=success)

                # Update state ensemble
                self.ensemble.enX = cp.deepcopy(self.enX_temp)
                self.ensemble.enX_temp = None

                # Update ensemble weights
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                success = False
                self.log_update(success=success)
                self.lam = self.lam * self.gamma
                # Increase damping parameter (divide calculations for ANALYSISDEBUG purpose)
                self.logger(f'Data misfit increased! λ increased: {self.lam / self.gamma} ──> {self.lam}')

            if not success:
                # Reset the objective function after report, so the next
                # comparison is against the last *accepted* misfit. The
                # per-realisation array is restored with it: update_step
                # reports that array, and the loop derives the scalars from
                # it, so leaving it holding the rejected attempt would put
                # them back out of step.
                self.data_misfit_mean = self.prev_data_misfit_mean
                self.data_misfit_std = self.prev_data_misfit_std
                if self.prev_ensemble_misfit is not None:
                    self.ensemble_misfit = self.prev_ensemble_misfit

            self._converged = False
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

    def log_update(self, success, prior_run=False):
        '''
        Log the update results in a formatted table.
        '''
        info = {
            "Iteration"     : f'{0 if prior_run else self.iteration + 1}',
            "Status"        : "Success" if (prior_run or success) else "Failed",
            "Data Misfit"   : self.data_misfit_mean,
            "Change (%)"    : '',
            "λ"             : self.lam
        }
        if not prior_run:
            delta = 100*(self.data_misfit_mean / self.prev_data_misfit_mean - 1)
            info["Change (%)"] = delta

        self.logger(**info)




#: Historical names.
lmenrmlMixIn = LMEnRML


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
    is divided by ``gamma_factor`` and the iteration retried.

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
        # AssimilationSchemeBase's `misfit_tol`/`step_tol` docs for why.
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

    def score_prior(self):
        """Score the prior forecast and fix the step length if left to 'auto'.

        See :meth:`LMEnRML.score_prior`; the same reasoning applies, with
        ``gamma`` in place of ``lam``.
        """
        self.enPred = self.pred_data.to_matrix()

        data_misfit = at.calc_objectivefun(self.enObs, self.enPred, self.cov_data)

        self.ensemble_misfit = data_misfit
        self.data_misfit_mean = np.mean(data_misfit)
        self.prior_data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        if self.gamma == 'auto':
            self.gamma = 0.1

        self.log_update(success=True, prior_run=True)

    def calc_analysis(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.

        """

        self.enPred = self.pred_data.to_matrix()

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
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
                self.ensemble.enX_temp = self.enX + self.gamma * self.step
            # Vector update following e.g. Evensen et al. 2019, for the
            # additive-anomaly flavours (subspace_update and friends).
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.gamma * self.w_step
                self.ensemble.enX_temp = np.dot(self.prior_enX, (np.eye(self.ne) + self.W / np.sqrt(self.ne - 1)))
            # Matrix update following e.g. Raanes et al. 2019, for flavours
            # that deliver a multiplicative ensemble-transform matrix instead
            # (margIS_update: W_0 = I, not the w_step branch's W_0 = 0).
            if hasattr(self, 'W_step'):
                self.W = self.current_W + self.gamma * self.W_step
                X_p = self.prior_enX @ self.proj * np.sqrt(self.ne - 1)
                self.ensemble.enX_temp = np.mean(self.prior_enX, axis=1, keepdims=True) + np.dot(X_p, self.W)

            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX.indices}
            self.ensemble.enX_temp.clip_matrix(limits)

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> StepReport:
        """Run one GN-EnRML step: analysis, forecast, then score and commit.

        Returns
        -------
        bool
            Whether the step was accepted. A rejected step leaves ``enX``
            untouched and backs off, so the loop retries at the same iteration
            number rather than advancing.
        """
        self.calc_analysis()
        self.after_analysis()
        self.run_forecast()
        self.score_and_commit()
        return StepReport(accepted=self.step_accepted, misfit=self.ensemble_misfit)

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
        enPred = self.pred_data.to_matrix()

        # Initialize the initial success value
        success = False

        self.prev_data_misfit_mean = self.data_misfit_mean
        self.prev_data_misfit_std = self.data_misfit_std
        self.prev_ensemble_misfit = getattr(self, "ensemble_misfit", None)

        data_misfit = at.calc_objectivefun(self.enObs, enPred, self.cov_data)
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
                self.log_update(success=success)
                self.logger.info(
                    f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                    f'from {self.prior_data_misfit_mean:0.1f} to {self.prev_data_misfit_mean:0.1f}')
            else:
                self.log_update(success=True)
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
                self.log_update(success=success)

                if self.gamma_factor > 1:
                    self.gamma = self.gamma + (self.gamma_max - self.gamma) * 2 ** (
                        -(self.iteration + 1) / (self.gamma_factor - 1)
                    )

                self.ensemble.enX = cp.deepcopy(self.enX_temp)
                self.ensemble.enX_temp = None
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            elif self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std >= self.prev_data_misfit_std:
                # accept itaration, but keep lam the same
                success = True
                self.log_update(success=success)

                self.ensemble.enX = cp.deepcopy(self.enX_temp)
                self.ensemble.enX_temp = None
                if hasattr(self, 'W'):
                    self.current_W = cp.deepcopy(self.W)

            else:  # Reject iteration, and increase lam
                success = False
                self.log_update(success=success)

                if self.gamma_factor > 1:
                    self.gamma = self.gamma / self.gamma_factor

                self.logger(
                    f'Data misfit increased! New Gamma for repeated analysis: {self.gamma}'
                )

            if not success:
                # Restore the last accepted misfit, per-realisation array
                # included -- update_step reports that array and the loop
                # derives the scalars from it.
                self.data_misfit_mean = self.prev_data_misfit_mean
                self.data_misfit_std = self.prev_data_misfit_std
                if self.prev_ensemble_misfit is not None:
                    self.ensemble_misfit = self.prev_ensemble_misfit

            self._converged = False
            self.step_accepted = success
            self.why_stop = why_stop
            return why_stop

    def log_update(self, success, prior_run=False):
        '''
        Log the update results in a formatted table.
        '''
        info = {
            "Iteration"     : f'{0 if prior_run else self.iteration + 1}',
            "Status"        : "Success" if (prior_run or success) else "Failed",
            "Data Misfit"   : self.data_misfit_mean,
            "Change (%)"    : '',
            "γ"             : self.gamma
        }
        if not prior_run:
            delta = 100 * (self.data_misfit_mean / self.prev_data_misfit_mean - 1)
            info["Change (%)"] = delta

        self.logger(**info)


#: Historical names.
gnenrmlMixIn = GNEnRML


class co_lm_enrml(LMEnRML, approx_update):
    """
    This is the implementation of the approximative LM-EnRML algorithm as described in [`chen2013`][].

    This algorithm is quite similar to the lm_enrml as provided above, and will therefore inherit most of its methods.
    We only change the calc_analysis part...

    % Copyright (c) 2019-2022 NORCE, All Rights Reserved. 4DSEIS
    """

    def __init__(self, keys_da):
        """Build the ensemble from the config and bind the analysis.

        See the class docstring for the parameters.
        """
        # Call __init__ in parent class
        super().__init__(keys_da)

    def calc_analysis(self):
        """
        Calculate the update step in approximate LM-EnRML code.

        Attributes
        ----------
        iteration : int
            Iteration number

        Returns
        -------
        success : bool
            True if data mismatch is decreasing, False if increasing
        """
        # Get assimilation order as a list
        self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

        # When handling large cases, it may be very costly to assemble the data covariance and localizaton matrix.
        # To alleviate this in the simultuaneus-iterative scheme we store these matrices, the list of states and
        # the list of data types after the first iteration.

        if not hasattr(self, 'list_datatypes'):
            # Get list of data types to be assimilated and of the free states. Do this once, because listing keys from a
            # Python dictionary just when needed (in different places) may not yield the same list!
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, self.assim_index)
            self.list_states = list(self.state.keys())

            # self.cov_data = np.load('CD.npz')['arr_0']
            # Generate the realizations of the observed data once
            # Augment observed and predicted data
            self.obs_data_vector, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                            self.list_datatypes)
            obs_data_vector = self.obs_data_vector

            # Generate the data auto-covariance matrix
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if hasattr(self, 'cov_data'):  # cd matrix has been imported
                    tmp_E = np.dot(cholesky(self.cov_data).T,
                                   np.random.randn(self.cov_data.shape[0], self.ne))
                else:
                    tmp_E = at.extract_tot_empirical_cov(
                        self.datavar, self.assim_index, self.list_datatypes, self.ne)
                # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    tmp_E = at.screen_data(tmp_E, self.aug_pred_data,
                                           obs_data_vector, self.iteration)
                self.E = tmp_E
                self.real_obs_data = obs_data_vector[:, np.newaxis] - tmp_E

                self.cov_data = np.var(self.E, ddof=1,
                                       axis=1)  # calculate the variance, to be used for e.g. data misfit calc
                # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
                self.scale_data = np.sqrt(self.cov_data)
            else:
                if not hasattr(self, 'cov_data'):  # if cd is not loaded
                    self.cov_data = at.gen_covdata(
                        self.datavar, self.assim_index, self.list_datatypes)
                # data screening
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    self.cov_data = at.screen_data(
                        self.cov_data, self.aug_pred_data, obs_data_vector, self.iteration)

                init_en = Cholesky()  # Initialize GeoStat class for generating realizations
                self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                       return_chol=True)

            self.datavar = at.update_datavar(
                self.cov_data, self.datavar, self.assim_index, self.list_datatypes)
            self.current_state = cp.deepcopy(self.state)

            # Calc. misfit for the initial iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data, self.aug_pred_data, self.cov_data)
            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit_mean = np.mean(data_misfit)
            self.prior_data_misfit_mean = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            if self.lam == 'auto':
                self.lam = 0.5 * self.prior_data_misfit_mean

        else:
            _, self.aug_pred_data = at.aug_obs_pred_data(
                self.obs_data, self.pred_data, self.assim_index, self.list_datatypes)

        # Mean pred_data and perturbation matrix with scaling
        mean_preddata = np.mean(self.aug_pred_data, 1)
        if len(self.scale_data.shape) == 1:
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * (
                    self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne))))
            else:
                pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * (
                    self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne)))) / \
                    (np.sqrt(self.ne - 1))
        else:
            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                pert_preddata = solve(self.scale_data, self.aug_pred_data -
                                      np.dot(mean_preddata[:, None], np.ones((1, self.ne))))
            else:
                pert_preddata = solve(self.scale_data, self.aug_pred_data - np.dot(mean_preddata[:, None], np.ones((1, self.ne)))) / \
                    (np.sqrt(self.ne - 1))
        self.pert_preddata = pert_preddata

        self.step = self.update()
        if self.step is not None:
            aug_state_upd = at.aug_state(self.current_state, self.list_states) + self.step
        if hasattr(self, 'w_step'):
            self.W = self.current_W - self.w_step
            aug_prior_state = at.aug_state(self.prior_state, self.list_states)
            aug_state_upd = np.dot(aug_prior_state, (np.eye(
                self.ne) + self.W / np.sqrt(self.ne - 1)))

        # Extract updated state variables from aug_update
        self.state = at.update_state(aug_state_upd, self.state, self.list_states)
        self.state = at.limits(self.state, self.prior_info)

class gn_enrml(LMEnRML):
    """
    This is the implementation of the stochastig IES as  described in [`raanes2019`][].

    More information about the method is found in [`evensen2019`][].
    This implementation is the Gauss-Newton version.

    This algorithm is quite similar to the `lm_enrml` as provided above, and will therefore inherit most of its methods.
    We only change the calc_analysis part...
    """

    def __init__(self, keys_da):
        """Build the ensemble from the config and bind the analysis.

        See the class docstring for the parameters.
        """
        # Call __init__ in parent class
        super().__init__(keys_da)

    def calc_analysis(self):
        """
        Changelog
        ---------
        - KF 25/2-20
        """
        # Get assimilation order as a list
        assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]

        # When handling large cases, it may be very costly to assemble the data covariance and localizaton matrix.
        # To alleviate this in the simultuaneus-iterative scheme we store these matrices, the list of states and
        # the list of data types after the first iteration.

        if not hasattr(self, 'list_datatypes'):
            # Get list of data types to be assimilated and of the free states. Do this once, because listing keys from a
            # Python dictionary just when needed (in different places) may not yield the same list!
            self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(
                self.obs_data, assim_index)
            self.list_states = list(self.state.keys())

            # Generate the realizations of the observed data once
            # Augment observed and predicted data
            self.obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                                   self.list_datatypes)
            obs_data_vector = self.obs_data_vector

            if 'emp_cov' in self.keys_da and self.keys_da['emp_cov'] == 'yes':
                if hasattr(self, 'cov_data'):  # cd matrix has been imported
                    tmp_E = np.dot(cholesky(self.cov_data).T, np.random.randn(
                        self.cov_data.shape[0], self.ne))
                else:
                    tmp_E = at.extract_tot_empirical_cov(
                        self.datavar, assim_index, self.list_datatypes, self.ne)
                # self.E = (tmp_E - tmp_E.mean(1)[:,np.newaxis])/np.sqrt(self.ne - 1)/
                self.real_obs_data = obs_data_vector[:, np.newaxis] - tmp_E

                self.cov_data = np.var(tmp_E, ddof=1,
                                       axis=1)  # calculate the variance, to be used for e.g. data misfit calc
                # self.cov_data = ((self.E * self.E)/(self.ne-1)).sum(axis=1) # calculate the variance, to be used for e.g. data misfit calc
                self.scale_data = np.sqrt(self.cov_data)
            else:
                if not hasattr(self, 'cov_data'):  # if cd is not loaded
                    self.cov_data = at.gen_covdata(
                        self.datavar, assim_index, self.list_datatypes)
                # data screening
                if 'screendata' in self.keys_da and self.keys_da['screendata'] == 'yes':
                    self.cov_data = at.screen_data(
                        self.cov_data, pred_data, obs_data_vector, self.iteration)

                init_en = Cholesky()  # Initialize GeoStat class for generating realizations
                self.real_obs_data, self.scale_data = init_en.gen_real(self.obs_data_vector, self.cov_data, self.ne,
                                                                       return_chol=True)

            self.datavar = at.update_datavar(
                self.cov_data, self.datavar, assim_index, self.list_datatypes)
            cov_data = self.cov_data
            obs_data = self.real_obs_data
            #
            self.current_state = cp.deepcopy(self.state)
            #
            self.aug_prior = cp.deepcopy(at.aug_state(
                self.current_state, self.list_states))
            # self.mean_prior = aug_prior.mean(axis=1)
            # self.X = (aug_prior - np.dot(np.resize(self.mean_prior, (len(self.mean_prior), 1)),
            #                                                  np.ones((1, self.ne))))
            self.W = np.zeros((self.ne, self.ne))

            self.proj = (np.eye(self.ne) - (1 / self.ne) *
                         np.ones((self.ne, self.ne))) / np.sqrt(self.ne - 1)
            self.E = np.dot(obs_data, self.proj)

            # Calc. misfit for the initial iteration
            if len(cov_data.shape) == 1:
                tmp_data_misfit = np.diag(np.dot((pred_data - obs_data).T,
                                                 np.dot(np.expand_dims(self.cov_data ** (-1), axis=1),
                                                        np.ones((1, self.ne))) * (pred_data - obs_data)))
            else:
                tmp_data_misfit = np.diag(
                    np.dot((pred_data - obs_data).T, solve(self.cov_data, (pred_data - obs_data))))
            mean_data_misfit = np.mean(tmp_data_misfit)
            # mean_data_misfit = np.median(tmp_data_misfit)
            std_data_misfit = np.std(tmp_data_misfit)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit_mean = mean_data_misfit
            self.prior_data_misfit_mean = mean_data_misfit
            self.data_misfit_std = std_data_misfit

        else:
            # for analysis debug...
            cov_data = self.cov_data
            obs_data_vector = self.obs_data_vector
            _, pred_data = at.aug_obs_pred_data(
                self.obs_data, self.pred_data, assim_index, self.list_datatypes)
            obs_data = self.real_obs_data

        if len(self.scale_data.shape) == 1:
            Y = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1), np.ones((1, self.ne))) * \
                np.dot(pred_data, self.proj)
        else:
            Y = solve(self.scale_data, np.dot(pred_data, self.proj))
        omega = np.eye(self.ne) + np.dot(self.W, self.proj)
        LU = lu_factor(omega.T)
        S = lu_solve(LU, Y.T).T
        if len(self.scale_data.shape) == 1:
            scaled_misfit = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                   np.ones((1, self.ne))) * (obs_data - pred_data)
        else:
            scaled_misfit = solve(self.scale_data, (obs_data - pred_data))

        u, s, v = np.linalg.svd(S, full_matrices=False)
        if self.trunc_energy < 1:
            ti = (np.cumsum(s) / sum(s)) <= self.trunc_energy
            u, s, v = u[:, ti].copy(), s[ti].copy(), v[ti, :].copy()

        ps_inv = np.diag([el_s ** (-1) for el_s in s])
        if len(self.scale_data.shape) == 1:
            X = np.dot(ps_inv, np.dot(u.T, np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                  np.ones((1, self.ne))) * self.E))
        else:
            X = np.dot(ps_inv, np.dot(u.T, solve(self.scale_data, self.E)))
        Lam, z = np.linalg.eig(np.dot(X, X.T))

        X2 = np.dot(u, np.dot(ps_inv.T, z))

        X3_m = np.dot(S.T, X2)
        # X3_old = np.dot(X2, np.linalg.solve(np.eye(len(Lam)) + np.diag(Lam), X2.T))
        step_m = np.dot(np.dot(X3_m, inv(np.eye(len(Lam)) + np.diag(Lam))),
                        np.dot(X3_m.T, self.W))

        if 'localization' in self.keys_da:
            if hasattr(self.localization, 'auto_ada_loc'):
                loc_step_d = np.dot(np.linalg.pinv(self.aug_prior), self.localization.auto_ada_loc(self.aug_prior,
                                                                                                   np.dot(np.dot(S.T, X2),
                                                                                                          np.dot(inv(
                                                                                                              np.eye(len(Lam)) + np.diag(Lam)),
                                                                                                       np.dot(X2.T, scaled_misfit))),
                                                                                                   self.list_states,
                                                                                                   **{'prior_info': self.prior_info}))
                self.step = self.lam * (self.W - (step_m + loc_step_d))
        else:
            step_d = np.dot(np.linalg.inv(omega).T,  np.dot(np.dot(Y.T, X2),
                                                            np.dot(inv(np.eye(len(Lam)) + np.diag(Lam)),
                                                                   np.dot(X2.T, scaled_misfit))))
            self.step = self.lam * (self.W - (step_m + step_d))

        self.W -= self.step

        aug_state_upd = np.dot(self.aug_prior, (np.eye(
            self.ne) + self.W / np.sqrt(self.ne - 1)))

        # Extract updated state variables from aug_update
        self.state = at.update_state(aug_state_upd, self.state, self.list_states)

        self.state = at.limits(self.state, self.prior_info)

    def check_convergence(self):
        """
        Check if GN-EnRML have converged based on evaluation of change sizes of objective function, state and damping
        parameter. Very similar to original function, but exit if there is no reduction in obj. function.

        Returns
        -------
        conv : bool
            Logic variable indicating if the algorithm has converged.

        status : bool
            Indicates whether the objective function has reduced.

        why_stop : dict
            Dictionary with keys corresponding to convergence criteria, with logical variables indicating
            which of them has been met.

        Changelog
        ---------
        - ST 3/6-16
        - ST 6/6-16: Added LM damping param. check
        - KF 16/11-20: Modified for GN-EnRML
        - KF 10/3-21: Output whether the method reduced the objective function
        """
        # Prelude to calc. conv. check (everything done below is from calc_analysis)
        if hasattr(self, 'list_datatypes'):
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes = self.list_datatypes
            cov_data = self.cov_data
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
        else:
            assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            list_datatypes, _ = at.get_list_data_types(self.obs_data, assim_index)
            # cov_data = at.gen_covdata(self.datavar, assim_index, list_datatypes)
            obs_data_vector, pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, assim_index,
                                                              list_datatypes)
            # mean_preddata = np.mean(pred_data, 1)

        success = False

        # if inital conv. check, there are no prev_data_misfit
        if self.prev_data_misfit_mean is None:
            self.data_misfit_mean = np.mean(self.data_misfit_mean)
            self.prev_data_misfit_mean = self.data_misfit_mean
            self.prev_data_misfit_std = self.data_misfit_std
            success = True
        # update the last mismatch, only if this was a reduction of the misfit
        if self.data_misfit_mean < self.prev_data_misfit_mean:
            self.prev_data_misfit_mean = self.data_misfit_mean
            self.prev_data_misfit_std = self.data_misfit_std
            success = True
        # if there was no reduction of the misfit, retain the old "valid" data misfit.

        # Calc. std dev of data misfit (used to update lamda)
        # mat_obs = np.dot(obs_data_vector.reshape((len(obs_data_vector), 1)), np.ones((1, self.ne)))  # use the perturbed
        # data instead.
        mat_obs = self.real_obs_data
        if len(cov_data.shape) == 1:
            data_misfit = np.diag(np.dot((pred_data - mat_obs).T,
                                         np.dot(np.expand_dims(self.cov_data ** (-1), axis=1),
                                                np.ones((1, self.ne))) * (pred_data - mat_obs)))
        else:
            data_misfit = np.diag(np.dot((pred_data - mat_obs).T,
                                  solve(self.cov_data, (pred_data - mat_obs))))
        self.data_misfit_mean = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)

        # # Calc. mean data misfit for convergence check, using the updated state variable
        # self.data_misfit_mean = np.dot((mean_preddata - obs_data_vector).T,
        #                      solve(cov_data, (mean_preddata - obs_data_vector)))
        # if self.data_misfit_mean > self.prev_data_misfit_mean:
        #    print(f'\n\nMisfit increased from {self.prev_data_misfit_mean:.1f} to {self.data_misfit_mean:.1f}. Exiting')
        #    self.logger.info(f'\n\nMisfit increased from {self.prev_data_misfit_mean:.1f} to {self.data_misfit_mean:.1f}. Exiting')

        # Convergence check: Relative step size of data misfit or state change less than tolerance
        if abs(1 - (self.data_misfit_mean / self.prev_data_misfit_mean)) < self.data_misfit_tol \
                or np.any(abs(np.mean(self.step, 1)) < self.step_tol) \
                or self.lam >= self.lam_max:
            # or self.data_misfit_mean > self.prev_data_misfit_mean:
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'step_size_stop': np.any(abs(np.mean(self.step, 1)) < self.step_tol),
                        'step_size': self.step,
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            if self.data_misfit_mean >= self.prev_data_misfit_mean:
                success = False
                self.logger.info(f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                                 f'from {self.prior_data_misfit_mean:0.1f} to {self.prev_data_misfit_mean:0.1f}')
            else:
                self.logger.info(f'Iterations have converged after {self.iteration + 1} iterations. Objective function reduced '
                                 f'from {self.prior_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}')

            # Return conv = True, why_stop var.
            return True, success, why_stop

        else:  # conv. not met
            # Logical variables for conv. criteria
            why_stop = {'data_misfit_stop': 1 - (self.data_misfit_mean / self.prev_data_misfit_mean) < self.data_misfit_tol,
                        'data_misfit': self.data_misfit_mean,
                        'prev_data_misfit': self.prev_data_misfit_mean,
                        'step_size': self.step,
                        'step_size_stop': np.any(abs(np.mean(self.step, 1)) < self.step_tol),
                        'lambda': self.lam,
                        'lambda_stop': self.lam >= self.lam_max}

            ###############################################
            ##### update Lambda step-size values ##########
            ###############################################
            if self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std < self.prev_data_misfit_std:
                # If reduction in mean data misfit, increase step length
                self.lam = self.lam + (self.lam_max - self.lam) * \
                    2 ** (-(self.iteration) / (self.gamma - 1))
                success = True
                self.current_state = cp.deepcopy(self.state)
            elif self.data_misfit_mean < self.prev_data_misfit_mean and self.data_misfit_std >= self.prev_data_misfit_std:
                # Accept itaration, but keep lam the same
                success = True
                self.current_state = cp.deepcopy(self.state)
            else:  # Reject iteration, and decrease step length
                self.lam = self.lam / self.gamma
                success = False

            if success:
                self.logger.info(f'Successfull iteration number {self.iteration}! Objective function reduced from '
                                 f'{self.prev_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}. New Lamba for next analysis: '
                                 f'{self.lam}')
            else:
                self.logger.info(f'Failed iteration number {self.iteration}! Objective function increased from '
                                 f'{self.prev_data_misfit_mean:0.1f} to {self.data_misfit_mean:0.1f}. New Lamba for repeated analysis: '
                                 f'{self.lam}')
                # Reset data misfit to prev_data_misfit (because the current state is neglected)
                self.data_misfit_mean = self.prev_data_misfit_mean
                self.data_misfit_std = self.prev_data_misfit_std

            return False, success, why_stop
