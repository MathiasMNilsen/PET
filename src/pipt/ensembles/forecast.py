"""Forecast support for assimilation ensembles.

Running the forward simulator and turning its raw output into ``pred_data`` is
ensemble work, not loop work: it reads ``sim``, ``enX``, ``data_df`` and the
compression machinery, and it writes ``pred_data``. It lived on
``pipt.loop.assimilation.Assimilate`` only because that class historically drove
every iteration.

:class:`AssimilationSchemeBase` expects its ensemble collaborator to expose a
public :meth:`ForecastMixin.forecast`, so the forecast lives here and the loop
delegates to it. Mixed into :class:`pipt.ensembles.AssimilationEnsemble`.
"""

import os
import pickle
from copy import deepcopy
from typing import Any

import numpy as np

import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract

__all__ = ["ForecastMixin", "OutlierMixin"]


class ForecastMixin:
    """Forward simulation and predicted-data preparation."""

    RESTART_RESULTS_FILE = "restart_sim_results.pkl"
    SIM_RESULTS_FILE = "sim_results.pkl"

    def forecast(self) -> None:
        """Run forecast simulations and prepare predicted data for analysis."""
        if self._load_restart_prediction_if_available():
            return

        enX = self.enX if self.enX_temp is None else self.enX_temp
        self.calc_prediction(enX)
        self.pred_data = self.sim_to_pred_data(self.sim_data)

        # Multilevel runs correct each level towards the reference level's mean.
        # This needs `pred_data`, so it happens here rather than inside
        # `calc_prediction`, which only produces `sim_data`.
        if getattr(self, "multilevel", None) is not None:
            self.treat_modeling_error()

        self._apply_prediction_scaling()

        if extract.is_enabled(self.keys_da.get("post_process_forecast", False)):
            self.post_process_forecast()

        self._save_forecast_debug()

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    @property
    def _saving_enabled(self) -> bool:
        return "nosave" not in self.keys_da

    @property
    def save_folder(self) -> str | None:
        """Folder for run artifacts, created on first use, or ``None``."""
        if not self._saving_enabled:
            return None
        folder = self.keys_da.get("savefolder", "Results")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _save_path(self, filename: str) -> str:
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        return os.path.join(self.save_folder, filename)

    # ------------------------------------------------------------------
    # Forecast steps
    # ------------------------------------------------------------------
    def _load_restart_prediction_if_available(self) -> bool:
        if not os.path.exists(self.RESTART_RESULTS_FILE):
            return False

        with open(self.RESTART_RESULTS_FILE, "rb") as file:
            self.sim_data = pickle.load(file)

        self.pred_data = self.sim_to_pred_data(self.sim_data)

        os.rename(self.RESTART_RESULTS_FILE, self.SIM_RESULTS_FILE)
        print("--- Restart sim results used ---")
        return True

    def _apply_prediction_scaling(self) -> None:
        if "scale" not in self.keys_da:
            return

        scale_keys, scale_factor = self.keys_da["scale"]
        for prediction in self.pred_data:
            for key in prediction:
                if key in scale_keys:
                    prediction[key] *= scale_factor

    def _save_forecast_debug(self) -> None:
        if "saveforecast" not in self.sim.input_dict:
            return
        if not self._saving_enabled:
            return

        forecast = self.sim_data
        if self.data_df.is_scaled:
            forecast = forecast.copy().invert_scale()

        with open(self._save_path(self.SIM_RESULTS_FILE), "wb") as file:
            pickle.dump(forecast, file)

    def sim_to_pred_data(self, pred: Any) -> Any:
        '''
        Filter the simulator output to match the structure of the predicted data expected.

        Parameters
        ----------
        pred : Any
            The raw output from the simulator, which may be a list of DataFrames or a single DataFrame.

        Returns
        -------
        Any
            The processed predicted data, structured to match the ensemble's expected format for analysis.
        '''
        if isinstance(pred, list):
            return [self.sim_to_pred_data(frame) for frame in pred]
        index = self.data_df.index
        columns = self.data_df.columns
        return pred.filter_dataframe(index=index, columns=columns)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def post_process_forecast(self) -> None:
        """Post-process predicted data after a forecast run."""
        compress_columns = self.sparse_info["compress_data"]
        if not isinstance(compress_columns, list):
            compress_columns = [compress_columns]
        pred_data_tmp = deepcopy(self.pred_data[compress_columns])

        self._apply_sim2seis_scaling(pred_data_tmp)
        self._apply_sparse_compression(pred_data_tmp)
        self._save_reconstructed_forecast_if_requested()

    def _apply_sim2seis_scaling(self, pred_data_tmp: Any) -> None:
        if not os.path.exists("scale_results.pkl"):
            return

        if self.scale_val is None:
            with open("scale_results.pkl", "rb") as file:
                scale = pickle.load(file)
            self.scale_val = np.sum(scale[0]) / len(scale[0])

        if self.sparse_info is not None:
            self._scale_sparse_sim2seis(pred_data_tmp, self.scale_val)
        else:
            self._scale_dense_sim2seis(self.scale_val)

    def _scale_sparse_sim2seis(self, pred_data_tmp: Any, scale_value: float) -> None:
        for index in pred_data_tmp.index:
            row = pred_data_tmp.loc[index]
            if row is None:
                continue
            for column in row:
                if "sim2seis" in column and row[column] is not None:
                    pred_data_tmp.at[index, column] = row[column] / scale_value

    def _scale_dense_sim2seis(self, scale_value: float) -> None:
        for index in self.pred_data.index:
            row = self.pred_data.loc[index]
            for column in row:
                if "sim2seis" in column and row[column] is not None:
                    self.pred_data.at[index, column] = row[column] / scale_value

    def _apply_sparse_compression(self, pred_data_tmp: Any) -> None:
        if not self.sparse_info:
            return

        self.data_rec = []
        compress_key = self.sparse_info["compress_data"]
        use_ensemble = self.sparse_info["use_ensemble"]
        ensemble_size = self.ne + 1 if self.keys_da["scheme"] == "gies" else self.ne

        vintage = 0
        for index in pred_data_tmp.index:
            cell = pred_data_tmp.loc[index, compress_key]
            if None in cell:
                continue

            data_length = len(self.data_df.loc[index, compress_key])
            self.pred_data.at[index, compress_key] = np.zeros((data_length, ensemble_size))

            for member in range(ensemble_size):
                compressed_data = self.compress_manager(
                    cell[:, member], vintage, use_ensemble,
                )
                self.pred_data.at[index, compress_key][:, member] = compressed_data
            vintage += 1

        if use_ensemble:
            self.compress_manager()
            self.sparse_info["use_ensemble"] = None

    def _save_reconstructed_forecast_if_requested(self) -> None:
        if "saveforecast" not in self.sim.input_dict:
            return
        if not self.sparse_data:
            return

        for vintage in np.arange(len(self.data_rec)):
            self.data_rec[vintage] = np.asarray(self.data_rec[vintage]).T

        with open("rec_results.pkl", "wb") as file:
            pickle.dump(self.data_rec, file)


class OutlierMixin:
    """Replacement of outlier ensemble members.

    Ensemble work, like the forecast: it rewrites ``pred_data``, ``sim_data``
    and the state matrix in place. Called between forecast and scoring, so the
    replacement feeds into the misfit the scheme sees.
    """

    def remove_outliers(self) -> None:
        """Replace outlier ensemble members with resampled non-outliers."""
        outlier_idx, non_outlier_idx = at.get_outlier_index(
            self.pred_data, self.data_df, self.data_var_df,
        )
        if len(outlier_idx) == 0:
            return
        idx = np.arange(self.ne)
        for outlier in outlier_idx:
            new_idx = np.random.choice(non_outlier_idx)
            idx[outlier] = new_idx
            self.logger(f"Replaced outlier {outlier} with member {new_idx}")

        # Remove outliers from state ensemble
        state_attribute = "enX_temp" if self.enX_temp is not None else "enX"
        enX_filtered = getattr(self, state_attribute)[:, idx]
        setattr(self, state_attribute, enX_filtered)

        # Filter outliers from dataframes
        def filter_outliers(cell):
            return cell[..., idx] if cell.ndim > 1 else cell[idx]
        self.pred_data = self.pred_data.map(filter_outliers)
        self.sim_data = self.sim_data.map(filter_outliers)
        if getattr(self, "adjoints", None) is not None:
            self.adjoints = self.adjoints.map(filter_outliers)
