"""Assimilation loop for iterative ensemble-based methods."""

import os
import pickle
import numpy as np
from copy import deepcopy
from importlib import import_module
from typing import Any

from pipt.loop.ensemble import Ensemble
from pipt.misc_tools import analysis_tools as at
from pipt.misc_tools.qaqc_tools import QAQC
import pipt.misc_tools.extract_tools as extract


class Assimilate:
    """Run iterative ensemble-based data assimilation.

    The loop supports the same responsibilities as the original implementation:

    * run prior and posterior forecasts,
    * call the ensemble update scheme through ``calc_analysis()``,
    * delegate convergence checks to ``check_convergence()``,
    * optionally run QA/QC, remove outliers, save debug artifacts and restart
      snapshots.

    The concrete assimilation mathematics remain in the ``Ensemble`` and update
    scheme classes; this class coordinates the workflow.
    """
    PRIOR_FORECAST_FILE = "prior_forecast.pkl"
    POSTERIOR_STATE_FILE = "posterior_state_estimate.npz"
    POSTERIOR_FORECAST_FILE = "posterior_forecast.pkl"
    RESTART_RESULTS_FILE = "restart_sim_results.pkl"
    SIM_RESULTS_FILE = "sim_results.pkl"
    STOP_REASON_FILE = "why_iter_loop_stopped.pkl"

    def __init__(self, ensemble: Ensemble):
        """Initialize the assimilation loop.

        Parameters
        ----------
        ensemble : Ensemble
            Prepared ensemble instance containing configuration, state,
            simulator, observations and update-scheme methods.
        """
        self.ensemble = ensemble
        self.max_iter = self._get_max_iterations()
        self.why_stop: dict[str, Any] | None = None
        self.qaqc: QAQC | None = None
        self.scale_val: float | None = None
        self.save_folder: str | None = None

        if self._saving_enabled:
            self.save_folder = self.ensemble.keys_da.get("savefolder", "Results")
            os.makedirs(self.save_folder, exist_ok=True)

    @property
    def _saving_enabled(self) -> bool:
        return "nosave" not in self.ensemble.keys_da

    def _get_max_iterations(self) -> int:
        if hasattr(self.ensemble, "max_iter"):
            return self.ensemble.max_iter
        return extract.extract_maxiter(self.ensemble.keys_da)

    def run(self) -> None:
        """Execute the full iterative assimilation workflow.

        The method coordinates the high-level data-assimilation loop while the
        ensemble/update-scheme object performs the algorithm-specific analysis
        and convergence calculations. The workflow is:

        1. Run a prior forecast at iteration zero.
        2. Optionally remove forecast/state outliers.
        3. Optionally run prior QA diagnostics.
        4. For each subsequent iteration, run ``calc_analysis()``, forecast the
           updated ensemble, remove outliers if configured, and call
           ``check_convergence()``.
        5. Persist configured iteration information, analysis-debug output,
           restart snapshots, final posterior estimates, and the final stopping
           reason.

        The loop stops when either the ensemble reaches ``self.max_iter`` or the
        update scheme reports convergence. Accepted iterations increment
        ``self.ensemble.iteration``; rejected iterations keep the same iteration
        number and allow the update scheme to retry according to its own state.

        Notes
        -----
        This method mutates the supplied ensemble in place. In particular,
        ``pred_data``, ``enX``, ``enX_temp``, ``iteration``, ``why_stop`` and
        optional diagnostic/restart files may be updated as part of the run.
        """
        converged = False
        self.qaqc = self._build_qaqc()

        while self.ensemble.iteration < self.max_iter and not converged:
            if self.ensemble.iteration == 0:
                self._run_prior_iteration()
                successful_iteration = True
            else:
                converged, successful_iteration = self._run_analysis_iteration()

            if successful_iteration:
                self._handle_successful_iteration()
                self.ensemble.iteration += 1

            if extract.is_enabled(self.ensemble.keys_da.get("restartsave", False)):
                self.ensemble.save()

        if self._saving_enabled:
            self._save_posterior_results()
            self._save_stop_reason(converged)
        self._log_convergence_summary()

    def _build_qaqc(self) -> QAQC | None:
        """Create QA/QC helper only when requested by the configuration."""
        qaqc_requested = (
            "qa" in self.ensemble.keys_da
            or "qa" in self.ensemble.sim.input_dict
            or "qc" in self.ensemble.keys_da
        )
        if not qaqc_requested:
            return None

        return QAQC(
            self.ensemble.keys_da | self.ensemble.sim.input_dict,
            self.ensemble.obs_data,
            self.ensemble.datavar,
            self.ensemble.logger,
            self.ensemble.prior_info,
            self.ensemble.sim,
            self.ensemble.prior_enX.to_dict(),
        )

    def _run_prior_iteration(self) -> None:
        """Forecast the prior ensemble and run optional prior QA."""
        self.calc_forecast()
        if "remove_outliers" in self.ensemble.keys_da:
            self._remove_outliers()
        self._run_prior_quality_assurance()
        self._save_prior_forecast()
        if "analysisdebug" in self.ensemble.keys_da:
            self._save_analysis_debug()

    def _run_prior_quality_assurance(self) -> None:
        if self.qaqc is None or "qa" not in self.ensemble.keys_da:
            return

        self.qaqc.set(
            self.ensemble.pred_data,
            self.ensemble.enX.to_dict(),
            self.ensemble.lam,
        )
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
        self.qaqc.calc_coverage()
        self.qaqc.calc_kg({"plot_all_kg": True, "only_log": False, "num_store": 5})

    def _save_prior_forecast(self) -> None:
        if not self._saving_enabled:
            return
        try:
            self.ensemble.pred_data.to_pickle(self._save_path(self.PRIOR_FORECAST_FILE))
        except Exception:
            np.savez(self._save_path(self.PRIOR_FORECAST_FILE), pred_data=self.ensemble.pred_data)

    def _run_analysis_iteration(self) -> tuple[bool, bool]:
        """Run analysis, forecast, outlier handling and convergence check."""
        self.ensemble.calc_analysis()
        self._refresh_screened_qaqc_datavar()

        self.calc_forecast()
        if "remove_outliers" in self.ensemble.keys_da:
            self._remove_outliers()

        converged, successful_iteration, self.why_stop = self.ensemble.check_convergence()
        return converged, successful_iteration

    def _refresh_screened_qaqc_datavar(self) -> None:
        """Update QAQC data variance after first-iteration data screening."""
        if self.qaqc is None:
            return
        if "qa" not in self.ensemble.keys_da:
            return
        if not extract.is_enabled(self.ensemble.keys_da.get("screendata", False)):
            return
        if self.ensemble.iteration != 1:
            return

        self.ensemble.logger.info("Recomputing Mahalanobis distance with updated datavar")
        self.qaqc.datavar = self.ensemble.datavar
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))

    def _handle_successful_iteration(self) -> None:
        """Persist iteration artifacts and run QA/QC after accepted updates."""
        if "iterinfo" in self.ensemble.keys_da:
            self._save_iteration_information()

        if self.ensemble.iteration == 0:
            return

        if "analysisdebug" in self.ensemble.keys_da:
            self._save_analysis_debug()

        if self.qaqc is None:
            return

        if "qc" in self.ensemble.keys_da:
            self.qaqc.set(
                self.ensemble.pred_data,
                self.ensemble.enX.to_dict(),
                self.ensemble.lam,
            )
            self.qaqc.calc_da_stat()

        if "qa" in self.ensemble.keys_da:
            self.qaqc.set(
                self.ensemble.pred_data,
                self.ensemble.enX.to_dict(),
                self.ensemble.lam,
            )
            self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
            self.qaqc.calc_kg()

    def _save_posterior_results(self) -> None:
        """Save posterior state and forecast, falling back to pickle if needed."""
        try:
            np.savez(self._save_path(self.POSTERIOR_STATE_FILE), **self.ensemble.enX.to_dict())
            self.ensemble.pred_data.to_pickle(self._save_path(self.POSTERIOR_FORECAST_FILE))
        except Exception:
            with open(self._save_path(self.POSTERIOR_STATE_FILE), "wb") as file:
                pickle.dump(self.ensemble.enX.to_dict(), file)
            with open(self._save_path(self.POSTERIOR_FORECAST_FILE), "wb") as file:
                pickle.dump(self.ensemble.pred_data, file)

    def _save_stop_reason(self, converged: bool) -> None:
        if converged:
            reason = "Convergence criteria met. Stopping assimilation loop."
            self.ensemble.logger.info(reason)
        else:
            reason = "Maximum iterations reached without convergence."
            self.ensemble.logger.info(reason)
            
        why = self.why_stop.copy() if isinstance(self.why_stop, dict) else self.why_stop
        if why is not None:
            why["conv_string"] = reason

        with open(self._save_path(self.STOP_REASON_FILE), "wb") as file:
            pickle.dump(why, file, protocol=4)

    def _log_convergence_summary(self) -> None:
        if self.ensemble.prev_data_misfit is None:
            return

        out_str = "\n Convergence was met."
        if self.ensemble.prior_data_misfit > self.ensemble.data_misfit:
            out_str += (
                f" Obj. function reduced from {self.ensemble.prior_data_misfit:0.1f} "
                f"to {self.ensemble.data_misfit:0.1f}"
            )
        self.ensemble.logger(out_str)

    def _save_path(self, filename: str) -> str:
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        return os.path.join(self.save_folder, filename)

    def _remove_outliers(self) -> None:
        """Remove outlier ensemble members from prediction and state data."""
        state_attribute = "enX_temp" if self.ensemble.enX_temp is not None else "enX"
        filtered_prediction, filtered_state = at.remove_outliers(
            self.ensemble.pred_data,
            self.ensemble.data_df,
            getattr(self.ensemble, state_attribute),
            self.ensemble.data_var_df,
        )
        self.ensemble.pred_data = filtered_prediction
        setattr(self.ensemble, state_attribute, filtered_state)

    def _save_iteration_information(self) -> None:
        """Run configured iteration-info hooks."""
        for element in self._as_list(self.ensemble.keys_da["iterinfo"]):
            if ".py" not in element:
                continue

            module_name = element.removesuffix(".py")
            iter_info_func = import_module(module_name)
            iter_info_func.main(self)

    def _save_analysis_debug(self) -> None:
        """Save requested analysis-debug variables."""
        save_dict: dict[str, Any] = {}

        for save_type in self._as_list(self.ensemble.keys_da["analysisdebug"]):
            if hasattr(self, save_type):
                save_dict[save_type] = getattr(self, save_type)
            elif hasattr(self.ensemble, save_type):
                if save_type == 'pred_data':
                    # Make tolist of records (dataframe cannot be saved in .npz file)
                    save_dict[save_type] = self.ensemble.pred_data.to_dict(orient='records')
                else:
                    save_dict[save_type] = getattr(self.ensemble, save_type)
            elif save_type == "state":
                save_dict.update(self._state_debug_dict())
            else:
                print(f"Cannot save {save_type}, because it is a local variable!\n\n")

        save_dict["savefolder"] = self.save_folder
        at.save_analysisdebug(self.ensemble.iteration, **save_dict)

    def _state_debug_dict(self) -> dict[str, Any]:
        if hasattr(self.ensemble, "multilevel") and self.ensemble.multilevel is not None:
            return {
                f"state_level{level}": self.ensemble.enX[level].to_dict()
                for level in range(self.ensemble.tot_level)
            }
        return self.ensemble.enX.to_dict()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    def calc_forecast(self) -> None:
        """Run forecast simulations and prepare predicted data for analysis."""
        if self._load_restart_prediction_if_available():
            return

        state = self.ensemble.enX if self.ensemble.enX_temp is None else self.ensemble.enX_temp
        self.ensemble.calc_prediction(enX=state)
        self.ensemble.pred_data = self.filter_pred_data(
            self.ensemble.data_df,
            self.ensemble.pred_data,
        )

        self._apply_prediction_scaling()

        if extract.is_enabled(self.ensemble.keys_da.get("post_process_forecast", False)):
            self.post_process_forecast()

        self._save_forecast_debug()

    def _load_restart_prediction_if_available(self) -> bool:
        if not os.path.exists(self.RESTART_RESULTS_FILE):
            return False

        with open(self.RESTART_RESULTS_FILE, "rb") as file:
            self.ensemble.pred_data = pickle.load(file)
        os.rename(self.RESTART_RESULTS_FILE, self.SIM_RESULTS_FILE)
        print("--- Restart sim results used ---")
        return True

    def _apply_prediction_scaling(self) -> None:
        if "scale" not in self.ensemble.keys_da:
            return

        scale_keys, scale_factor = self.ensemble.keys_da["scale"]
        for prediction in self.ensemble.pred_data:
            for key in prediction:
                if key in scale_keys:
                    prediction[key] *= scale_factor

    def _save_forecast_debug(self) -> None:
        if "saveforecast" not in self.ensemble.sim.input_dict:
            return
        if not self._saving_enabled:
            return

        forecast = self.ensemble.pred_data
        if self.ensemble.data_df.is_scaled:
            forecast = forecast.copy().invert_scale()

        with open(self._save_path(self.SIM_RESULTS_FILE), "wb") as file:
            pickle.dump(forecast, file)

    def filter_pred_data(self, data_df: Any, pred_df: Any) -> Any:
        """Filter predicted data to observed indices and columns.

        Parameters
        ----------
        data_df : pandas.DataFrame-like
            Observed data frame.
        pred_df : pandas.DataFrame-like or list[pandas.DataFrame-like]
            Predicted data frame(s) to filter.

        Returns
        -------
        pandas.DataFrame-like or list[pandas.DataFrame-like]
            Prediction data aligned to ``data_df``.
        """
        if isinstance(pred_df, list):
            return [self.filter_pred_data(data_df, frame) for frame in pred_df]

        if data_df.index.dtype == pred_df.index.dtype:
            pred_df = pred_df[pred_df.index.isin(data_df.index)]
        elif data_df.index.size != pred_df.index.size:
            raise ValueError("Index of pred_data and data_df do not match in type or size!")

        pred_df = pred_df[data_df.columns]
        if pred_df.empty:
            raise ValueError("No matching indices between pred_data and data_df after filtering!")

        return pred_df

    def post_process_forecast(self) -> None:
        """Post-process predicted data after a forecast run."""
        pred_data_tmp = deepcopy(self.ensemble.pred_data[self.ensemble.sparse_info["compress_data"]])

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

        if self.ensemble.sparse_info is not None:
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
        for index in self.ensemble.pred_data.index:
            row = self.ensemble.pred_data.loc[index]
            for column in row:
                if "sim2seis" in column and row[column] is not None:
                    self.ensemble.pred_data.at[index, column] = row[column] / scale_value

    def _apply_sparse_compression(self, pred_data_tmp: Any) -> None:
        if not self.ensemble.sparse_info:
            return

        self.ensemble.data_rec = []
        compress_key = self.ensemble.sparse_info["compress_data"]
        use_ensemble = self.ensemble.sparse_info["use_ensemble"]
        ensemble_size = self.ensemble.ne + 1 if self.ensemble.keys_da["daalg"][1] == "gies" else self.ensemble.ne

        for vintage, (index, row) in enumerate(pred_data_tmp.iterrows()):
            if row is None or compress_key not in row:
                continue

            data_length = len(self.ensemble.data_df.loc[index, compress_key])
            self.ensemble.pred_data.at[index, compress_key] = np.zeros((data_length, ensemble_size))

            for member in range(pred_data_tmp.loc[index, compress_key].shape[1]):
                compressed_data = self.ensemble.compress_manager(
                    pred_data_tmp.loc[index, compress_key][:, member],
                    vintage,
                    use_ensemble,
                )
                self.ensemble.pred_data.at[index, compress_key][:, member] = compressed_data

        if use_ensemble:
            self.ensemble.compress_manager()
            self.ensemble.sparse_info["use_ensemble"] = None

    def _save_reconstructed_forecast_if_requested(self) -> None:
        if "saveforecast" not in self.ensemble.sim.input_dict:
            return
        if not self.ensemble.sparse_data:
            return

        for vintage in np.arange(len(self.ensemble.data_rec)):
            self.ensemble.data_rec[vintage] = np.asarray(self.ensemble.data_rec[vintage]).T

        with open("rec_results.pkl", "wb") as file:
            pickle.dump(self.ensemble.data_rec, file)
