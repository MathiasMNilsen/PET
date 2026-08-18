"""
ES-MDA type schemes
"""

# External imports
import scipy.linalg as scilinalg
from copy import deepcopy
import numpy as np
from geostat.decomp import Cholesky

# Internal imports
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.scheme_base import AssimilationSchemeBase
from pipt.update_schemes.workflow import AssimilationWorkflowMixin
from pipt.update_schemes.strategy import StrategyMixin
import pipt.misc_tools.analysis_tools as at

# Flavours are resolved through the strategy registry now, not mixed in.

__all__ = [
    'ESMDA',
    'esmda_approx',
    'esmda_full',
    'esmda_subspace',
    'esmda_geo'
]

class ESMDA(AssimilationWorkflowMixin, StrategyMixin, AssimilationSchemeBase):
    """Ensemble Smoother with Multiple Data Assimilation (ES-MDA).

    An iterative ensemble smoother that assimilates all data repeatedly over a
    fixed number of steps, inflating the data-error covariance at each one so
    that the repeated conditioning does not over-fit. With inflation factors
    :math:`\\alpha_i` satisfying :math:`\\sum_i 1/\\alpha_i = 1`, each step applies

    .. math::

        m \\leftarrow m + C_{md} (C_{dd} + \\alpha_i C_d)^{-1} (d_{obs} - g(m))

    with the observations re-perturbed as
    :math:`d_{obs} = d_{true} + \\sqrt{\\alpha_i} C_d^{1/2} Z`.

    The schedule is fixed rather than convergence-driven, so a run normally
    ends by exhausting its steps and reports ``success=False``. That is the
    expected outcome, not a failure.

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
        simulator. Attribute reads the scheme does not own fall through to it,
        so ``scheme.enX`` and ``scheme.keys_da`` resolve as expected.
    strategy : pipt.update_schemes.analysis.AnalysisStrategy
        The bound analysis flavour.
    iteration : int
        Accepted iterations completed so far.
    data_misfit, prior_data_misfit : float
        Current and initial mean data misfit.

    Notes
    -----
    Configured through the ``mda`` block of ``keys_da``:

    ``tot_assim_steps``
        Number of assimilation steps, e.g. ``3``.
    ``inflation_param``
        Inflation factors, one per step, e.g. ``[3, 3, 3]``. Their reciprocals
        must sum to 1, which is asserted at construction. Defaults to
        ``tot_assim_steps`` repeated, which satisfies the constraint.

    Examples
    --------
    >>> result = ESMDA.assimilate(keys_da, keys_en, flow(keys_sim))
    >>> result.nit
    3

    References
    ----------
    Emerick and Reynolds, *Ensemble smoother with multiple data assimilation*
    [`emerick2013a`][]. For the geometric inflation schedule used by
    :class:`esmda_geo`, see Rafiee and Reynolds [`rafiee2017`][].

    See Also
    --------
    ES : Single-step smoother; ES-MDA with one assimilation step.
    LMEnRML : Iterates to convergence instead of on a fixed schedule.
    """

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        """Build the ensemble from the config and bind the analysis strategy.

        See the class docstring for the parameters.
        """
        # Build the collaborator, then hand it to the scheme base. Logging stays
        # on the ensemble's logger so the log output is unchanged.
        ensemble = Ensemble(keys_da, keys_en, sim)
        # misfit_tol/step_tol disable the base class's *generic* convergence
        # criteria. PIPT schemes decide convergence themselves, in
        # check_convergence(); letting the generic ones also fire would stop a
        # run early on a criterion the scheme never opted into.
        super().__init__(ensemble, logit=False, misfit_tol=0.0, step_tol=0.0)
        self.logger = ensemble.logger

        # The analysis flavour is a parameter of the algorithm, not a different
        # algorithm, so it selects a strategy object rather than a class.
        self.bind_strategy(self.resolve_analysis(analysis, keys_da))

        self.prev_data_misfit = None

        if self.restart is False:
            self.ensemble.prior_enX = deepcopy(self.enX)
            self.ensemble.list_states = list(self.enX.indices)
            self.ensemble.list_datatypes = self.keys_da['datatype']

            # At the moment, the iterative loop is threated as an iterative smoother an thus we check if assim. indices
            # are given as in the Simultaneous loop.
            #self.check_assimindex_simultaneous()
            #self.assim_index = [self.keys_da['obsname'], self.keys_da['assimindex'][0]]
            #self.list_datatypes, self.list_act_datatypes = at.get_list_data_types(self.obs_data, self.assim_index)

            # Extract no. assimilation steps from MDA keyword in DATAASSIM part of init. file and set this equal to
            # the number of iterations pluss one. Need one additional because the iter=0 is the prior run.
            self.max_iter = len(self._ext_assim_steps())+1
            # Prior forecast is not a counted iteration under the base loop.
            self.maxiter = self.max_iter - 1
            self.iteration = 0
            # Mirrored so ensemble-side helpers that consult the iteration
            # counter (e.g. data screening in perturb_observations) agree with
            # the scheme's, which is the one the loop advances.
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

            # Get state scaling and svd of scaled prior
            self.ensemble._ext_scaling()

        # Extract the inflation parameter from MDA keyword
        self.alpha = self._ext_inflation_param()

        self.prev_data_misfit = None

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> bool:
        """Run one ES-MDA assimilation step.

        Computes the inflated analysis, forecasts the trial state, then scores
        the resulting misfit and promotes the state. Scoring after the forecast
        is what lets outlier replacement, which runs in between, feed into the
        number the scheme sees.

        Returns
        -------
        bool
            Always ``True``. ES-MDA takes a fixed number of inflated steps and
            never rejects one. The ``success`` flag it logs compares the misfit
            against the previous iteration and is a *reporting* signal only --
            returning it here would make the base class discard accepted steps.
        """
        self.calc_analysis()
        self.after_analysis()
        self.run_forecast()
        self.score_and_commit()
        return True

    def check_convergence(self) -> bool:
        """ES-MDA runs its full schedule of inflated steps; nothing stops early."""
        return False

    def calc_analysis(self):
        r"""
        Analysis step of ES-MDA. The analysis algorithm is similar to EnKF analysis, only difference is that the data
        covariance matrix is inflated with an inflation parameter alpha. The update is done as an iterative smoother
        where all data is assimilated at once.

        Notes
        -----
        ES-MDA is an iterative ensemble smoother with a predefined number of iterations, where the updates is done with
        the EnKF update equations but where the data covariance matrix have been inflated:

        $$ \begin{align}
        d_{obs} &= d_{true} + \sqrt{\alpha}C_d^{1/2}Z \\
        m &= m_{prior} + C_{md}(C_g + \alpha C_d)^{-1}(g(m) - d_{obs})
        \end{align} $$

        where $d_{true}$ is the true observed data, $\alpha$ is the inflation factor, $C_d$ is the data covariance
        matrix, $Z$ is a standard normal random variable, $C_{md}$ and $C_{g}$ are sample covariance matrices,
        $m$ is the model parameter, and $g(\)$ is the predicted data. Note that $\alpha$ can have a different
        value in each assimilation step and must fulfill:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        where $N_a$ being the total number of assimilation steps.
        """
        # Get Ensemble matrix of predicted data
        self.enPred = self.pred_data.to_matrix()

        if self.iteration == 0:  # first iteration

            # Calculate the prior data misfit
            data_misfit = at.calc_objectivefun(
                self.enObs_conv,
                self.enPred,
                Cd=self.cov_data
            )
            #data_misfit = at.data_mismatch(self.vecObs, self.enPred, self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.prior_data_misfit = np.mean(data_misfit)
            self.prior_data_misfit_std = np.std(data_misfit)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)
            self.ensemble_misfit = data_misfit

            # Log initial data misfit
            self.log_update(prior_run=True)
            self.data_random_state = deepcopy(np.random.get_state())

            self.enObs, self.scale_data = Cholesky().gen_real(
                self.vecObs,
                self.alpha[self.iteration] * self.cov_data,
                self.ne,
                return_chol=True
            )
            self.E = np.dot(self.enObs, self.proj)

        else:
            self.data_random_state = deepcopy(np.random.get_state())
            self.enObs, self.scale_data = Cholesky().gen_real(
                self.vecObs,
                self.alpha[self.iteration] * self.cov_data,
                self.ne,
                return_chol=True
            )
            self.E = np.dot(self.enObs, self.proj)

        if 'localanalysis' in self.keys_da:
            self.ensemble.local_analysis_update()
        else:

            # Check for adjoint
            if hasattr(self, 'adjoints'):
                enAdj = self.adjoints.to_matrix(is_jacobian=True) # Shape (nd, nx, ne)
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

            # Update the state ensemble and weights. These land on the ensemble
            # explicitly: the forecast reads enX_temp off the collaborator, and
            # attribute delegation covers reads only.
            if self.step is not None:
                self.ensemble.enX_temp = self.enX + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                self.ensemble.enX_temp = np.dot(self.prior_enX, (np.eye(self.ne) + self.W/np.sqrt(self.ne - 1)))


            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.enX.indices}
            self.ensemble.enX_temp.clip_matrix(limits)

    def score_and_commit(self):
        """Score the forecast that followed the analysis, then commit the step.

        Was the second half of ``check_convergence``: ES-MDA never actually
        tested for convergence there, it recomputed the misfit, logged the
        iteration and promoted ``enX_temp``. Under the new contract the
        convergence question lives in :meth:`check_convergence` and this keeps
        the bookkeeping.

        Returns
        -------
        dict
            The ``why_stop`` record, also stored on ``self.why_stop``.
        """

        self.prev_data_misfit = self.data_misfit
        self.prev_data_misfit_std = self.data_misfit_std

        # Get Ensemble of predicted data
        enPred = self.pred_data.to_matrix()

        data_misfit = at.calc_objectivefun(self.enObs_conv, enPred, self.cov_data)
        self.data_misfit     = np.mean(data_misfit)
        self.data_misfit_std = np.std(data_misfit)
        self.ensemble_misfit = data_misfit

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        # Log update results
        success = self.data_misfit < self.prev_data_misfit
        self.log_update(success=success)

        # Promote the trial state. Written through the ensemble so the next
        # forecast and any external reader see it.
        self.ensemble.enX = deepcopy(self.enX_temp)
        self.ensemble.enX_temp = None
        if hasattr(self, 'W'):
            self.current_W = deepcopy(self.W)

        self.why_stop = why_stop
        return why_stop

    def log_update(self, success=None, prior_run=False):
        '''
        Log the update results in a formatted table.
        '''
        info = {
            "Iteration"     : f'{0 if prior_run else self.iteration + 1}',
            "Status"        : "Success" if (prior_run or success) else "Failed",
            "Data Misfit"   : self.data_misfit,
            "Change (%)"    : '',
            "α"             : self.alpha[self.iteration] if not prior_run else '',
        }
        if not prior_run:
            delta = 100*(self.data_misfit / self.prev_data_misfit - 1)
            info["Change (%)"] = delta

        self.logger(**info)

    def _ext_inflation_param(self):
        r"""
        Extract the data covariance inflation parameter from the MDA keyword in DATAASSIM part. Also, we check that
        the criterion:

        $$ \sum_{i=1}^{N_a} \frac{1}{\alpha} = 1 $$

        is fulfilled for the inflation factor, alpha. If the keyword for inflation parameter -- INFLATION_PARAM -- is
        not provided, we set the default $\alpha_i = N_a$, where $N_a$ is the tot. no. of MDA assimilation steps (the
        criterion is fulfilled with this value).

        Returns
        -------
        alpha: list
            Data covariance inflation factor
        """
        try:
            mda_opts = dict(self.keys_da['mda'])
        except Exception:
            mda_opts = dict([self.keys_da['mda']])

        # Check if INFLATION_PARAM has been provided, and if so, extract the value(s). If not, we set alpha to the
        # default value equal to the tot. no. assim. steps
        if 'inflation_param' in mda_opts:
            alpha_tmp = mda_opts['inflation_param']
            alpha = alpha_tmp if isinstance(alpha_tmp, list) else [alpha_tmp] * len(self._ext_assim_steps())

            assert len(alpha) == len(self._ext_assim_steps()), \
            'Number of INFLATION_PARAM values does not match TOT_ASSIM_STEPS!'
        else:
            n_steps = len(self._ext_assim_steps())
            alpha = [n_steps] * n_steps

        # Check if alpha fulfills the criterion to machine precision
        assert 1 - np.finfo(float).eps <= sum(1/x for x in alpha) <= 1 + np.finfo(float).eps, \
            'Sum of inverse inflation parameters does not add up to 1!'

        return alpha

    def _ext_assim_steps(self):
        """
        Extract list of assimilation steps to perform in MDA loop from the MDA keyword (mandatory for
        MDA class) in DATAASSIM part. (This method is similar to Iterative._ext_max_iter)

        Parameters
        ----------
        keys_da : dict
            all keywords from DATAASSIM part
        mda : info
            for MDA methods

        Returns
        -------
        int
            Total number of MDA assimilation steps

        Changelog
        ---------
        - ST 7/6-16
        - ST 1/3-17: Changed to output list of assim. steps instead of just tot. assim. steps
        """
        try:
            mda_opts = dict(self.keys_da['mda'])
        except Exception:
            mda_opts = dict([self.keys_da['mda']])


        # Check if 'max_iter' has been given; if not, give error (mandatory in ITERATION)
        try:
            assim_steps = list(range(int(mda_opts['tot_assim_steps'])))
        except KeyError:
            raise AssertionError('TOT_ASSIM_STEPS has not been given in MDA!')

        # If it is a restart run, we remove simulations already done
        if self.restart is True:
            # List simulations we already have done. Do this by checking pred_data.
            # OBS: Minus 1 here do to the aborted simulation is also not None.
            # TODO: Relying on loop_ind may not be the best strategy (?)
            sim_done = list(range(self.loop_ind))

            # Update list of assim. steps by removing simulations we have done
            assim_steps = [ind for ind in assim_steps if ind not in sim_done]

        # Return list assim. steps
        return assim_steps


#: Historical name. ``multilevel.esmda_hybrid`` still subclasses it.
esmdaMixIn = ESMDA


class esmda_approx(ESMDA):
    """Deprecated alias: prefer ``ESMDA(..., analysis="approx")``."""

    FLAVOUR = "approx"


class esmda_full(ESMDA):
    """Deprecated alias: prefer ``ESMDA(..., analysis="full")``."""

    FLAVOUR = "full"


class esmda_subspace(ESMDA):
    """Deprecated alias: prefer ``ESMDA(..., analysis="subspace")``."""

    FLAVOUR = "subspace"


class esmda_geo(esmda_approx):
    """
    This is the implementation of the ES-MDA-GEO algorithm from [1]. The main analysis step in this algorithm is the
    same as the standard ES-MDA algorithm (implemented in the `es_mda` class). The difference between this and the
    standard algorithm is the calculation of the inflation factor. Also see [`rafiee2017`][].
    """

    def __init__(self, keys_da):
        """Build the ensemble from the config and bind the analysis strategy.

        See the class docstring for the parameters.
        """
        # Pass the init_file upwards in the hierarchy
        super().__init__(keys_da)

        # Within
        self.alpha = [None] * self.tot_assim

    def _calc_inflation_factor(self, pert_preddata, cov_data, energy=99):
        """
        We calculate the inflation factor, follow the procedure laid out in Algorithm 1 in [1].

        Parameters
        ----------
        pert_preddata : ndarray
            Predicted data (fwd. run) ensemble matrix perturbed with its mean
        cov_data : ndarray
            Data covariance matrix
        energy : float, optional
            Percentage of energy kept in (T)SVD decompostion of 'sensitivity' matrix (default is 99%)

        Returns
        -------
        alpha : float
            Inflation factor
        beta : float
            Geometric factor
        """
        # Need the square-root of the data covariance matrix
        if np.count_nonzero(cov_data - np.diagonal(cov_data)) == 0:
            l = np.sqrt(cov_data)  # only variance (diagonal) term
        else:
            # Cholesky decomposition
            l = scilinalg.cholesky(cov_data)  # cov. matrix has off-diag. terms

        # Calculate the 'sensitivity' matrix:
        sens = (1 / np.sqrt(self.ne - 1)) * np.dot(l, pert_preddata)

        # Perform SVD on sensitivtiy matrix
        _, s_d, _ = np.linalg.svd(sens, full_matrices=False)

        # If no. measurements is more than ne - 1, we only keep ne - 1 sing. val.
        if sens.shape[0] >= self.ne:
            s_d = s_d[:-1].copy()

        # If energy is less than 100 we truncate the SVD matrices
        if energy < 100:
            ti = (np.cumsum(s_d) / sum(s_d)) * 100 <= energy
            s_d = s_d[ti].copy()

        # Calc average singular value
        avg_s_d = s_d.mean()

        # The inflation factor is chosen as the maximum of the average singular value (squared) and max. no. of
        # iterations
        alpha = np.max((avg_s_d ** 2, self.tot_assim))

        # We calculate the geometric (reduction) factor (called 'common ratio' in the article). The formula is given
        # as (1 - beta**-n) / (1 - beta**-1) = alpha (it is actually incorrect in the article, and should be as
        # written here), with n=tot. assim. steps. Rewritten:
        #
        # (1-alpha)*beta**n + alpha*beta**(n-1) - 1 = 0
        #
        # This is of course a nasty polynomial root problem, but we use Numpy.roots, extract the real
        # root less than 1, and hope for the best :p
        root_coeff = np.zeros(self.tot_assim + 1)
        root_coeff[0] = 1 - alpha  # first coeff. in polynomial
        root_coeff[1] = alpha  # sec. coeff in polynomial
        root_coeff[-1] = -1
        roots = np.roots(root_coeff)

        # Most likely the first root will be 1, and the second one will be the one we want. Due to numerical
        # imprecision, the first root will not be exactly one, so we us Numpy.min to get the second root.
        beta = np.min([x.real for x in roots if x.imag == 0 and x.real < 1])

        # Return inflation and geometric factor
        return alpha, beta
