"""Descriptive description."""

# External import
import os.path
import numpy as np
import sys
from copy import deepcopy, copy
from scipy.linalg import solve, cholesky
from scipy.spatial import distance
import itertools
from geostat.decomp import Cholesky

# Internal import
from ensemble.ensemble import Ensemble as PETEnsemble
from ensemble.logger import PetLogger
import misc.read_input_csv as rcsv
from pipt.misc_tools import wavelet_tools as wt
from pipt.misc_tools.cov_regularization import localization, _calc_distance
from misc.structures import PETDataFrame

# Import internal tools
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract


class Ensemble(PETEnsemble):
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

            - daalg: spesification of the method, first the main type (e.g., "enrml"), then the solver (e.g., "gnenrml")
            - analysis: update flavour ("approx", "full" or "subspace")
            - energy: percent of singular values kept after SVD
            - obsvarsave: save the observations as a file (default false)
            - restart: restart optimization from a restart file (default false)
            - restartsave: save a restart file after each successful iteration (defalut false)
            - analysisdebug: specify which class variables to save to the result files
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
        super(Ensemble, self).__init__(keys_da|keys_en, sim)

        # Setup logger
        self.logger = PetLogger(filename='assim.log')
        self.logger(f'=========== Running Data Assimilation - {keys_da["daalg"][0].upper()} ===========')

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
                self.localization = localization(
                    self.keys_da['localization'],
                    self.keys_da['truedataindex'],
                    self.keys_da['datatype'],
                    self.keys_en['state'],
                    self.ne
                )

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


    def compress_manager(self, data=None, vintage=0, aug_coeff=None):
        """
        Compress the input data using wavelets.

        Parameters
        ----------
        data : 
            data to be compressed
            If data is `None`, all data (true and simulated) is re-compressed (used if leading indices are updated)
        vintage : int
            the time index for the data
        aug_coeff : bool
            - False: in this case the leading indices for wavelet coefficients are computed
            - True: in this case the leading indices are augmented using information from the ensemble
            - None: in this case simulated data is compressed
        """

        # If input data is None, we re-compress all data
        data_array = None
        if data is None:
            vintage = 0
            for idx in self.data_df.index:  # TRUEDATAINDEX
                for col in self.data_df.columns:  # DATATYPE
                    data_array = self.data_df.loc[idx, col]

                    # Perform compression if required
                    if (data_array is not None) and (col in self.sparse_info['compress_data']):
                        data_array, wdec_rec = self.sparse_data[vintage].compress(data_array)  # compress
                        self.data_df.at[idx, col] = data_array  # save array in obs_data
                        rec = self.sparse_data[vintage].reconstruct(wdec_rec)  # reconstruct the data
                        np.savez('truedata_rec_' + str(vintage) + '.npz', rec)  # save reconstructed data
                        est_noise = np.power(self.sparse_data[vintage].est_noise, 2)
                        self.data_var_df.at[idx, col] = est_noise

                        # Update the ensemble
                        data_sim = self.pred_data.loc[idx, col]
                        self.pred_data.at[idx, col] = np.zeros((len(data_array), self.ne))
                        self.data_rec.append([])
                        for m in range(self.pred_data.at[idx, col].shape[1]):
                            data_array = data_sim[:, m]
                            data_array, wdec_rec = self.sparse_data[vintage].compress(data_array)  # compress
                            self.pred_data.at[idx, col][:, m] = data_array
                            rec = self.sparse_data[vintage].reconstruct(wdec_rec)  # reconstruct the data
                            self.data_rec[vintage].append(rec)

                        # Go to next vintage
                        vintage = vintage + 1

            del data_array # free memory

            # Option to store the dictionaries containing observed data and data variance
            if 'obsvarsave' in self.keys_da and self.keys_da['obsvarsave'] == 'yes':
                self.data_df.to_pickle('obs_data.pkl')
                self.data_var_df.to_pickle('obs_var.pkl')

            if 'saveforecast' in self.keys_en:
                s = 'prior_forecast_rec.npz'
                np.savez(s, self.data_rec)

        elif aug_coeff is None: # compress predicted data

            data_array, wdec_rec = self.sparse_data[vintage].compress(data)  # compress
            rec = self.sparse_data[vintage].reconstruct(
                wdec_rec)  # reconstruct the simulated data
            if len(self.data_rec) == vintage:
                self.data_rec.append([])
            self.data_rec[vintage].append(rec)

        # DEPRICATED!!!!
        #elif not aug_coeff: # compress true data, aug_coeff = false
        #
        #    options = copy(self.sparse_info)
        #    # find the correct mask for the vintage
        #    options['mask'] = options['mask'][vintage]
        #    if isinstance(options['min_noise'], list):
        #        if 0 <= vintage < len(options['min_noise']):
        #            options['min_noise'] = options['min_noise'][vintage]
        #        else:
        #            print('Error: min_noise must either be scalar or list with one number for each vintage')
        #            sys.exit(1)

        #    x = wt.SparseRepresentation(options)
        #    data_array, wdec_rec = x.compress(data, self.sparse_info['th_mult'])
        #    self.sparse_data.append(x)  # store the information
        #    data_rec = x.reconstruct(wdec_rec)  # reconstruct the data
        #    s = 'truedata_rec_' + str(vintage) + '.npz'
        #    np.savez(s, data_rec)  # save reconstructed data
        #    if self.sparse_info['use_ensemble']:
        #        data_array = data  # just return the same as input
        
        elif aug_coeff:

            _, _ = self.sparse_data[vintage].compress(data, self.sparse_info['th_mult'])
            data_array = data  # just return the same as input

        return data_array

    def local_analysis_update(self):
        '''
        Function for updates that can be used by all algorithms. Do this once to avoid duplicate code for local
        analysis.
        '''
        # Copy original info to restore after local updates
        orig_list_data = deepcopy(self.list_datatypes)
        orig_list_state = deepcopy(self.list_states)
        orig_cd = deepcopy(self.cov_data)
        orig_real_obs_data = deepcopy(self.real_obs_data)
        orig_data_vector = deepcopy(self.obs_data_vector)

        # loop over the states that we want to update. Assume that the state and data combinations have been
        # determined by the initialization.
        # TODO: augment parameters with identical mask.

        # REGION PARAMETERS
        ############################################################################################################
        for state in self.local_analysis['region_parameter']:
            self.list_datatypes = [
                elem for elem in self.list_datatypes if
                elem in self.local_analysis['update_mask'][state]
            ]
            self.list_states = [deepcopy(state)]

            self._ext_scaling()  # scaling for this state
            if 'localization' in self.keys_da:
                self.localization.loc_info['field'] = self.state_scaling.shape
            del self.cov_data

            # reset the random state for consistency
            np.random.set_state(self.data_random_state)
            self.vecObs, self.enObs = self.set_observations()
            _, self.enPred = at.aug_obs_pred_data(
                self.obs_data, 
                self.pred_data, 
                self.assim_index,
                self.list_datatypes
            )

            # Get state ensemble for list_states
            enX = []
            idX = {}
            for idx in self.list_states:
                start, end = self.idX[idx]
                tempX = self.enX[start:end, :]
                enX.append(tempX)
                idX[idx] = (enX.shape[0] - tempX.shape[0], enX.shape[0])

            # Compute the analysis update
            self.update(
                enX = np.vstack(enX),
                enY = self.enPred,
                enE = self.enObs,
            )

            # Update the state
            if hasattr(self, 'step'):
                self.enX_temp = self.enX + self.step
        ############################################################################################################

        # VECTOR REGION PARAMETERS
        ############################################################################################################
        for state in self.local_analysis['vector_region_parameter']:
            current_list_datatypes = deepcopy(self.list_datatypes)
            for state_indx in range(self.state[state].shape[0]): # loop over the elements in the region
                self.list_datatypes = [elem for elem in self.list_datatypes if
                                       elem in self.local_analysis['update_mask'][state][state_indx]]
                if len(self.list_datatypes):
                    self.list_states = [deepcopy(state)]
                    self._ext_scaling()  # scaling for this state
                    if 'localization' in self.keys_da:
                        self.localization.loc_info['field'] = self.state_scaling.shape
                    del self.cov_data
                    # reset the random state for consistency
                    np.random.set_state(self.data_random_state)
                    self._ext_obs()  # get the data that's in the list of data.
                    _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data, self.assim_index,
                                                                 self.list_datatypes)
                    # Mean pred_data and perturbation matrix with scaling
                    if len(self.scale_data.shape) == 1:
                        self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                    np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
                    else:
                        self.pert_preddata = solve(
                            self.scale_data, np.dot(self.aug_pred_data, self.proj))

                    aug_state = at.aug_state(self.current_state, self.list_states)[state_indx,:]
                    self.update()
                    if hasattr(self, 'step'):
                        aug_state_upd = aug_state + self.step[state_indx,:]
                    self.state[state][state_indx,:] = aug_state_upd

                self.list_datatypes = deepcopy(current_list_datatypes)
        ############################################################################################################


        for state in self.local_analysis['cell_parameter']:
            self.list_states = [deepcopy(state)]
            self._ext_scaling()  # scaling for this state
            orig_state_scaling = deepcopy(self.state_scaling)
            param_position = self.local_analysis['parameter_position'][state]
            field_size = param_position.shape
            for k in range(field_size[0]):
                for j in range(field_size[1]):
                    for i in range(field_size[2]):
                        current_data_list = list(
                            self.local_analysis['update_mask'][state][k][j][i])
                        current_data_list.sort()  # ensure consistent ordering of data
                        if len(current_data_list):
                            # if non-unique data for assimilation index, get the relevant data.
                            if self.local_analysis['unique'] == False:
                                orig_assim_index = deepcopy(self.assim_index)
                                assim_index_data_list = set(
                                    [el.split('_')[0] for el in current_data_list])
                                current_assim_index = [
                                    int(el.split('_')[1]) for el in current_data_list]
                                current_data_list = list(assim_index_data_list)
                                self.assim_index[1] = current_assim_index
                            self.list_datatypes = deepcopy(current_data_list)
                            del self.cov_data
                            # reset the random state for consistency
                            np.random.set_state(self.data_random_state)
                            self._ext_obs()
                            _, self.aug_pred_data = at.aug_obs_pred_data(self.obs_data, self.pred_data,
                                                                         self.assim_index,
                                                                         self.list_datatypes)
                            # get parameter indexes
                            full_cell_index = np.ravel_multi_index(
                                np.array([[k], [j], [i]]), tuple(field_size))
                            # count active values
                            self.cell_index = [sum(param_position.flatten()[:el])
                                               for el in full_cell_index]
                            if 'localization' in self.keys_da:
                                self.localization.loc_info['field'] = (
                                    len(self.cell_index),)
                                self.localization.loc_info['distance'] = _calc_distance(
                                    self.local_analysis['data_position'],
                                    self.local_analysis['unique'],
                                    current_data_list, self.assim_index,
                                    self.obs_data, self.pred_data, [(k, j, i)])
                            # Set relevant state scaling
                            self.state_scaling = orig_state_scaling[self.cell_index]

                            # Mean pred_data and perturbation matrix with scaling
                            if len(self.scale_data.shape) == 1:
                                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data,
                                                                                            self.proj)
                            else:
                                self.pert_preddata = solve(
                                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

                            aug_state = at.aug_state(
                                self.current_state, self.list_states, self.cell_index)
                            self.update()
                            if hasattr(self, 'step'):
                                aug_state_upd = aug_state + self.step
                            self.state = at.update_state(
                                aug_state_upd, self.state, self.list_states, self.cell_index)

                            if self.local_analysis['unique'] == False:
                                # reset assim index
                                self.assim_index = deepcopy(orig_assim_index)
                            if hasattr(self, 'localization') and 'distance' in self.localization.loc_info:  # reset
                                del self.localization.loc_info['distance']

        self.list_datatypes = deepcopy(orig_list_data)  # reset to original list
        self.list_states = deepcopy(orig_list_state)
        self.cov_data = deepcopy(orig_cd)
        self.real_obs_data = deepcopy(orig_real_obs_data)
        self.obs_data_vector = deepcopy(orig_data_vector)
        self.cell_index = None
