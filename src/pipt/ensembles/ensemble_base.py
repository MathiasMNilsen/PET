"""Ensemble container for ensemble-based data assimilation.

The PIPT counterpart to :mod:`popt.ensembles.ensemble_base`: holds the state
realisations, observed data, localization and forward simulator for an
assimilation run.

Previously ``pipt.loop.ensemble.Ensemble``; that module has been removed.
"""

import os.path

import numpy as np
from scipy.linalg import cholesky
from geostat.decomp import Cholesky

from ensemble import BaseEnsemble, NullLogger, PetLogger
import misc.read_input_csv as rcsv
from pipt.localization import build_localization_instance
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

from pipt.ensembles.compression import CompressionMixin
from pipt.ensembles.forecast import ForecastMixin, OutlierMixin
from pipt.ensembles.local_analysis import LocalAnalysisMixin

__all__ = ["AssimilationEnsemble"]


class NoLocalization:
    """Stands in for a localization when the config asks for none.

    Analyses branch on ``localization.name``; ``None`` means no localization.
    A module-level class rather than an anonymous one so the ensemble that
    holds it can be pickled -- ``emergency_dump`` and the restart file both
    pickle the ensemble, and an anonymous class made that fail exactly when
    a run had crashed.
    """

    name = None


class AssimilationEnsemble(ForecastMixin, OutlierMixin, CompressionMixin, LocalAnalysisMixin, BaseEnsemble):
    """
    Class for organizing/initializing misc. variables and simulator for an
    ensemble-based inversion run. Inherits the PET ensemble structure
    """

    def __init__(self, keys_da, keys_en, sim):
        """
        Parameters
        ----------
        keys_da : dict
            Options for the data assimilation class

            - scheme: name of the assimilation algorithm (e.g., "esmda", "lmenrml", "gnenrml")
            - analysis: update flavour ("approx", "full" or "subspace")
            - energy: percent of singular values kept after SVD
            - obsvarsave: save the observations as a file (default false)
            - restart: restart optimization from a restart file (default false)
            - restartsave: save a restart file after each successful iteration (defalut false)
            - savedata: names of scheme attributes to write to one file per
              iteration, ``assimilation_result_{i}.npz``. Iteration 0 is the
              prior. ``"state"`` expands to one array per state variable;
              anything else is looked up on the scheme and then on the
              ensemble, so e.g. ``"ensemble_misfit"``, ``"pred_data"``,
              ``"data_misfit"`` and ``"lam"`` all resolve. A name that resolves
              nowhere is reported and skipped. Omitting the key disables the
              saving, so there is no separate on/off switch.
              (Was ``analysisdebug``, still honoured with a warning.)
            - savefolder (or save_folder): where run artifacts go
              (default ``Results``)
            - logit: enable run logging (default true). When false, no log
              file is created and self.logger(...) calls become no-ops.
            - logger_name: log file name (default ``ASSIM.log``)
            - nosave: present in the config disables artifact saving entirely
            - truedataindex: order of the simulated data (for timeseries this is points in time)
            - obsname: unit for truedataindex (for timeseries this is days or hours or seconds, etc.)
            - truedata: the data, e.g., provided as a .csv file
            - assimindex: index for the data that will be used for assimilation
            - datatype: list with the name of the datatypes
            - staticvar: name of the static variables
            - dynamicvar: name of the dynamic variables
            - datavar: data variance, e.g., provided as a .csv file

        keys_en : dict
            Options for the ensemble class

            - ne: number of perturbations used to compute the gradient
            - state: name of state variables passed to the .mako file
            - prior_<name>: the prior information the state variables, including mean, variance and variable limits

            NB: If keys_en is empty dict, it is assumed that the prior info is contained in keys_da.
            The merged dict keys_da|keys_en is what is sent to the parent class.

        sim : callable
            The forward simulator (e.g. flow)
        """


        # do the initiallization of the PETensemble
        super().__init__(keys_da | keys_en, sim)

        # Setup logger. logit=False replaces it with a no-op so every scheme's
        # unconditional self.logger(...) calls stay valid without a file being
        # created.
        if keys_da.get('logit', True):
            self.logger = PetLogger(filename=keys_da.get('logger_name', 'ASSIM.log'))
        else:
            self.logger = NullLogger()
        self.logger(f'=========== Running Data Assimilation - {keys_da["scheme"].upper()} ===========')

        # Internalize PIPT dictionary
        if not hasattr(self, 'keys_da'):
            self.keys_da = keys_da
        if not hasattr(self, 'keys_en'):
            self.keys_en = keys_en

        if self.restart is False:
            # Init in _init_prediction_output (used in run_prediction)
            self.prediction = None
            self.temp_state = None  # temporary state saving
            self.cov_prior = None  # Prior cov. matrix
            self.sparse_info = None  # Init in _org_sparse_representation
            self.sparse_data = []  # List of the compression info
            self.data_rec = []  # List of reconstructed data
            self.scale_val = None  # Use to scale data

            # Prepare sparse representation
            if 'compress' in self.keys_da:
                self.sparse_info = extract.organize_sparse_representation(self.keys_da['compress'])
            else:
                self.sparse_info = None

            # Load the data
            reader = rcsv.DataReader(self.keys_da, sparse_info=self.sparse_info)
            self.data_df = reader.get_data()
            self.sparse_data = reader.sparse_data
            self.data_var_df = reader.get_variance(self.data_df, reader.sparse_data)

            if self.keys_da.get('scale_data', False):
                self.data_df.scale('max-min')

                if self.keys_da.get('emp_cov', False):
                    self.data_var_df.scale('max-min',
                            minimum=self.data_df.scale_min,
                            maximum=self.data_df.scale_max,
                    )
                else:
                    self.data_var_df.scale('max-min',
                            minimum=0,
                            maximum=(self.data_df.scale_max - self.data_df.scale_min)**2
                    )

            self.keys_da['datatype'] = reader.datatype
            self.keys_da['truedataindex'] = reader.truedataindex
            self.keys_da['assimindex'] = reader.assimindex

            #self._org_obs_data() # Depricated!!
            #self._org_data_var() # Depricated!!

            # Define projection operator for centring and scaling ensemble matrix
            self.proj = (np.eye(self.ne) - np.ones((self.ne, self.ne))/self.ne) / np.sqrt(self.ne - 1)

            # Option to store the dictionaries containing observed data and data variance
            if extract.is_enabled(self.keys_da.get('obsvarsave', False)):
                # Save data_df and data_var_df as pickle files
                folder = self.keys_da.get('savefolder', './')
                # Check if folder exists, if not create it
                if not os.path.exists(folder):
                    os.makedirs(folder)
                self.data_df.to_pickle(f'{folder}/obs_data.pkl')
                self.data_var_df.to_pickle(f'{folder}/obs_var.pkl')

            # Initialize localization
            if 'localization' in self.keys_da:
                self.localization = build_localization_instance(
                    self.keys_da['localization'],
                    self.keys_da['truedataindex'],
                    self.keys_da['datatype'],
                    self.keys_en['state'],
                    self.ne,
                    data=self.data_df,
                    prior_info=self.prior_info,
                )
            else:
                self.localization = NoLocalization()

            # Initialize local analysis
            if 'localanalysis' in self.keys_da:
                self.local_analysis = extract.extract_local_analysis_info(self.keys_da['localanalysis'], self.idX.keys())

            self.pred_data  = None  # predicted data or forward simulation
            self.cell_index = None  # default value for extracting states

    def check_assimindex_simultaneous(self):
        """
        Check if assim. indices is given as a 1D list as is needed in simultaneous updating. If not, make it a 2D list
        with one row.
        """
        # Check if ASSIMINDEX is a list. If not, make it a 2D list with one row
        if not isinstance(self.keys_da['assimindex'], list):
            self.keys_da['assimindex'] = [[self.keys_da['assimindex']]]

        # Check if ASSIMINDEX is a 1D list. If true, make it a 2D list with one row
        elif not isinstance(self.keys_da['assimindex'][0], list):
            self.keys_da['assimindex'] = [self.keys_da['assimindex']]

        # If ASSIMINDEX is a 2D list, we reshape it to a 2D list with one row
        elif isinstance(self.keys_da['assimindex'][0], list):
            self.keys_da['assimindex'] = [
                [item for sublist in self.keys_da['assimindex'] for item in sublist]]

    def perturb_observations(self, vecObs):
        '''
        Generate the perturbed observed data ensemble
        '''
        # Generate ensemble of perturbed observed data
        if extract.is_enabled(self.keys_da.get('emp_cov', False)):
            if hasattr(self, 'cov_data'):  # cd matrix has been imported
                # enObs: samples from N(0,Cd)
                enObs = cholesky(self.cov_data).T @ np.random.randn(self.cov_data.shape[0], self.ne)
            else:
                enObs = self.data_var_df.to_matrix()

            # Screen data if required
            if extract.is_enabled(self.keys_da.get('screendata', False)):
                enObs = at.screen_data(
                    enObs,
                    self.enPred,
                    vecObs,
                    self.iteration
                )

            # Center the ensemble of perturbed observed data
            # enObs = vecObs[:, np.newaxis] - enObs
            self.cov_data = np.var(enObs, ddof=1, axis=1)
            self.scale_data = np.sqrt(self.cov_data)

        else:
            if not hasattr(self, 'cov_data'):  # if cd is not loaded
                cov = at.construct_data_cov(self.data_var_df)
                self.cov_data = cov[~np.isnan(cov)]

            # data screening
            if extract.is_enabled(self.keys_da.get('screendata', False)):
                self.cov_data = at.screen_data(
                    data = self.cov_data,
                    aug_pred_data = self.enPred,
                    obs_data_vector = vecObs,
                    iteration = self.iteration
                )

            generator = Cholesky()  # Initialize GeoStat class for generating realizations
            enObs, self.scale_data = generator.gen_real(
                mean = vecObs,
                var = self.cov_data,
                number = self.ne,
                return_chol = True
            )

        return enObs

    def _ext_scaling(self):
        # get vector of scaling
        self.state_scaling = at.calc_scaling(
            self.prior_enX, self.prior_enX.indices, self.prior_info)

        self.Am = None
