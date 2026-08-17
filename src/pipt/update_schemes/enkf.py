"""
EnKF type schemes
"""
# External imports
import numpy as np
from copy import deepcopy
from geostat.decomp import Cholesky                     # Making realizations

# Internal imports
from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.scheme_base import AssimilationSchemeBase
from pipt.update_schemes.workflow import AssimilationWorkflowMixin
from pipt.update_schemes.strategy import StrategyMixin
# Misc. tools used in analysis schemes
from pipt.misc_tools import analysis_tools as at
import pipt.misc_tools.ensemble_tools as entools
import pipt.misc_tools.extract_tools as extract



class EnKF(AssimilationWorkflowMixin, StrategyMixin, AssimilationSchemeBase):
    """
    Straightforward EnKF analysis scheme implementation. The sequential updating can be done with general grouping and
    ordering of data. If only one-step EnKF is to be done, use `es` instead.
    """

    def __init__(self, keys_da, keys_en, sim, analysis=None):
        """
        The class is initialized by passing the PIPT init. file upwards in the hierarchy to be read and parsed in
        `pipt.input_output.pipt_init.ReadInitFile`.
        """
        # Build the collaborator, then hand it to the scheme base. Logging
        # stays on the ensemble's logger so log output is unchanged.
        ensemble = Ensemble(keys_da, keys_en, sim)
        # misfit_tol/step_tol disable the base class's *generic* convergence
        # criteria. PIPT schemes decide convergence themselves, in
        # check_convergence(); letting the generic ones also fire would stop a
        # run early on a criterion the scheme never opted into.
        super().__init__(ensemble, logit=False, misfit_tol=0.0, step_tol=0.0)
        self.logger = ensemble.logger

        # Flavour is a parameter, so it selects a strategy object not a class.
        self.bind_strategy(self.resolve_analysis(analysis, keys_da))

        self.prev_data_misfit = None

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

    def calc_analysis(self):
        """
        Calculate the analysis step of the EnKF procedure. The updating is done using the Kalman filter equations, using
        svd for numerical stability. Localization is available.
        """
        # If this is initial analysis we calculate the objective function for all data. In the final convergence check
        # we calculate the posterior objective function for all data
        if self.prior_data_misfit is None:
            enPred = self.pred_data.to_matrix()

            # Calc. misfit for the initial iteration
            data_misfit = at.calc_objectivefun(self.enObs, enPred, self.scale_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}.')

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
                self.ensemble.enX_temp = self.enX + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                self.ensemble.enX_temp = np.dot(self.prior_enX, (np.eye(self.ne) + self.W/np.sqrt(self.ne - 1)))

            # Ensure limits are respected
            limits = {key: self.prior_info[key].get('limits', (None, None)) for key in self.idX.keys()}
            self.ensemble.enX_temp = entools.clip_matrix(self.enX_temp, limits, self.idX)

    # ------------------------------------------------------------------
    # AssimilationSchemeBase contract
    # ------------------------------------------------------------------
    def update_step(self) -> bool:
        """Run one EnKF step: analysis, forecast, then score and commit.

        Returns
        -------
        bool
            Always ``True``. The EnKF applies one update per data group and
            has no rejection path.
        """
        self.calc_analysis()
        self.after_analysis()
        self.run_forecast()
        self.score_and_commit()
        return True

    def check_convergence(self) -> bool:
        """The EnKF runs its full sweep of data groups; nothing stops early."""
        return False

    def score_and_commit(self):
        """
        Calculate the "convergence" of the method. Important to
        """
        self.prev_data_misfit = self.prior_data_misfit

        # only calulate for the final (posterior) estimate
        if self.iteration + 1 == len(self.keys_da['assimindex']):
            enPred = self.pred_data.to_matrix()
            data_misfit = at.calc_objectivefun(self.enObs, enPred, self.scale_data)
            self.data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

        else:  # sequential updates not finished. Misfit is not relevant
            self.data_misfit = self.prior_data_misfit

        # Logical variables for conv. criteria
        why_stop = {'rel_data_misfit': 1 - (self.data_misfit / self.prev_data_misfit),
                    'data_misfit': self.data_misfit,
                    'prev_data_misfit': self.prev_data_misfit}

        # Update state ensemble
        self.ensemble.enX = deepcopy(self.enX_temp)
        self.ensemble.enX_temp = None

        if self.data_misfit == self.prev_data_misfit:
            self.logger.info(
                f'EnKF update {self.iteration} complete!')
        else:
            if self.data_misfit < self.prior_data_misfit:
                self.logger.info(
                    f'EnKF update complete! Objective function decreased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
            else:
                self.logger.info(
                    f'EnKF update complete! Objective function increased from {self.prior_data_misfit:0.1f} to {self.data_misfit:0.1f}.')
        self.why_stop = why_stop
        return why_stop


#: Historical name, kept for subclasses outside this module.
enkfMixIn = EnKF


class enkf_approx(EnKF):
    """Deprecated alias: prefer ``EnKF(..., analysis="approx")``."""

    FLAVOUR = "approx"


class enkf_full(EnKF):
    """Deprecated alias: prefer ``EnKF(..., analysis="approx")``.

    The EnKF does not iterate, so the standard scheme is always applied; this
    name resolves to the same "approx" strategy it always did.
    """

    FLAVOUR = "approx"


class enkf_subspace(EnKF):
    """Deprecated alias: prefer ``EnKF(..., analysis="subspace")``."""

    FLAVOUR = "subspace"
