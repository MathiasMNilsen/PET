"""Sparse-representation (wavelet compression) support for assimilation ensembles.

Split out of the ensemble class so the data container is not also carrying the
compression plumbing. Mixed into :class:`pipt.ensembles.AssimilationEnsemble`.
"""

import numpy as np

__all__ = ["CompressionMixin"]


class CompressionMixin:
    """Wavelet compression of simulated/observed data."""

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
