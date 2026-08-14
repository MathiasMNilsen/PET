"""Assimilation loop for iterative ensemble-based methods."""

import os
import pickle
import numpy as np
import pandas as pd
from importlib import import_module
from typing import Any

from pipt.ensembles import AssimilationEnsemble as Ensemble
from pipt.update_schemes.scheme_base import AssimilationSchemeBase
from pipt.misc_tools import analysis_tools as at
from pipt.misc_tools.qaqc_tools import QAQC
from misc.structures import PETDataFrame
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
    STOP_REASON_FILE = "why_iter_loop_stopped.pkl"

    #: Moved to the ensemble along with the forecast; aliased so any external
    #: reference to ``Assimilate.SIM_RESULTS_FILE`` keeps resolving.
    RESTART_RESULTS_FILE = Ensemble.RESTART_RESULTS_FILE
    SIM_RESULTS_FILE = Ensemble.SIM_RESULTS_FILE

    def __init__(self, ensemble: Ensemble):
        """Initialize the assimilation loop.

        Parameters
        ----------
        ensemble : Ensemble
            Prepared ensemble instance containing configuration, state,
            simulator, observations and update-scheme methods.
        """
        # A migrated scheme *has* an ensemble; a legacy one *is* one. Keeping
        # both handles lets this loop drive either while the migration is in
        # progress -- for a legacy scheme the two names point at one object.
        self.scheme = ensemble
        self.ensemble = getattr(ensemble, "ensemble", ensemble)
        self.new_style = isinstance(ensemble, AssimilationSchemeBase)
        self.max_iter = self._get_max_iterations()
        self.why_stop: dict[str, Any] | None = None
        self.qaqc: QAQC | None = None
        self.save_folder: str | None = None

        if self._saving_enabled:
            self.save_folder = self.ensemble.keys_da.get("savefolder", "Results")
            os.makedirs(self.save_folder, exist_ok=True)

    @property
    def _saving_enabled(self) -> bool:
        return "nosave" not in self.ensemble.keys_da

    def _get_max_iterations(self) -> int:
        if hasattr(self.scheme, "max_iter"):
            return self.scheme.max_iter
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

        while self.scheme.iteration < self.max_iter and not converged:
            if self.scheme.iteration == 0:
                self._run_prior_iteration()
                successful_iteration = True
            else:
                converged, successful_iteration = self._run_analysis_iteration()

            if successful_iteration:
                self._handle_successful_iteration()
                self.scheme.iteration += 1
                self.ensemble.iteration = self.scheme.iteration

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
            self.scheme.lam,
        )
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
        self.qaqc.calc_coverage()
        self.qaqc.calc_kg({"plot_all_kg": True, "only_log": False, "num_store": 5})

    def _save_prior_forecast(self) -> None:
        if not self._saving_enabled:
            return
        try:
            self.ensemble.sim_data.to_pickle(self._save_path(self.PRIOR_FORECAST_FILE))
        except Exception:
            np.savez(self._save_path(self.PRIOR_FORECAST_FILE), sim_data=self.ensemble.sim_data)

    def _run_analysis_iteration(self) -> tuple[bool, bool]:
        """Run analysis, forecast, outlier handling and convergence check.

        The interleaving is what matters and is identical for both scheme
        styles: analysis produces a trial state, the forecast runs on it, any
        outliers are replaced, and only then is the misfit scored -- so outlier
        replacement still feeds into the number the scheme sees.
        """
        self.scheme.calc_analysis()
        self._refresh_screened_qaqc_datavar()

        self.calc_forecast()
        if "remove_outliers" in self.ensemble.keys_da:
            self._remove_outliers()

        if self.new_style:
            # Scoring and the convergence question are separate under the new
            # contract. `step_accepted` carries the LM family's rejection: a
            # rejected step leaves enX uncommitted and must not advance the
            # iteration counter, which is exactly what returning False here does.
            self.why_stop = self.scheme.score_and_commit()
            return self.scheme.check_convergence(), self.scheme.step_accepted

        converged, successful_iteration, self.why_stop = self.scheme.check_convergence()
        return converged, successful_iteration

    def _refresh_screened_qaqc_datavar(self) -> None:
        """Update QAQC data variance after first-iteration data screening."""
        if self.qaqc is None:
            return
        if "qa" not in self.ensemble.keys_da:
            return
        if not extract.is_enabled(self.ensemble.keys_da.get("screendata", False)):
            return
        if self.scheme.iteration != 1:
            return

        self.ensemble.logger.info("Recomputing Mahalanobis distance with updated datavar")
        self.qaqc.datavar = self.ensemble.datavar
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))

    def _handle_successful_iteration(self) -> None:
        """Persist iteration artifacts and run QA/QC after accepted updates."""
        if "iterinfo" in self.ensemble.keys_da:
            self._save_iteration_information()

        if self.scheme.iteration == 0:
            return

        if "analysisdebug" in self.ensemble.keys_da:
            self._save_analysis_debug()

        if self.qaqc is None:
            return

        if "qc" in self.ensemble.keys_da:
            self.qaqc.set(
                self.ensemble.pred_data,
                self.ensemble.enX.to_dict(),
                self.scheme.lam,
            )
            self.qaqc.calc_da_stat()

        if "qa" in self.ensemble.keys_da:
            self.qaqc.set(
                self.ensemble.pred_data,
                self.ensemble.enX.to_dict(),
                self.scheme.lam,
            )
            self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
            self.qaqc.calc_kg()

    def _save_posterior_results(self) -> None:
        """Save posterior state and forecast, falling back to pickle if needed."""
        try:
            np.savez(self._save_path(self.POSTERIOR_STATE_FILE), **self.ensemble.enX.to_dict())
            self.ensemble.sim_data.to_pickle(self._save_path(self.POSTERIOR_FORECAST_FILE))
        except Exception:
            with open(self._save_path(self.POSTERIOR_STATE_FILE), "wb") as file:
                pickle.dump(self.ensemble.enX.to_dict(), file)
            with open(self._save_path(self.POSTERIOR_FORECAST_FILE), "wb") as file:
                pickle.dump(self.ensemble.sim_data, file)

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
        if self.scheme.prev_data_misfit is None:
            return

        out_str = "\n Convergence was met."
        if self.scheme.prior_data_misfit > self.scheme.data_misfit:
            out_str += (
                f" Obj. function reduced from {self.scheme.prior_data_misfit:0.1f} "
                f"to {self.scheme.data_misfit:0.1f}"
            )
        self.ensemble.logger(out_str)

    def _save_path(self, filename: str) -> str:
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        return os.path.join(self.save_folder, filename)

    def _remove_outliers(self) -> None:
        """Remove outlier ensemble members from simulation and state data."""
        outlier_idx, non_outlier_idx = at.get_outlier_index(
            self.ensemble.pred_data, self.ensemble.data_df, self.ensemble.data_var_df,
        )
        if len(outlier_idx) == 0:
            return
        idx = np.arange(self.ensemble.ne)
        for outlier in outlier_idx:
            new_idx = np.random.choice(non_outlier_idx)
            idx[outlier] = new_idx
            self.ensemble.logger(f"Replaced outlier {outlier} with member {new_idx}")

        # Remove outliers from state ensemble
        state_attribute = "enX_temp" if self.ensemble.enX_temp is not None else "enX"
        enX_filtered = getattr(self.ensemble, state_attribute)[:, idx]
        setattr(self.ensemble, state_attribute, enX_filtered)

        # Filter outliers from dataframes
        def filter_outliers(cell):
            return cell[..., idx] if cell.ndim > 1 else cell[idx]
        self.ensemble.pred_data = self.ensemble.pred_data.map(filter_outliers)
        self.ensemble.sim_data = self.ensemble.sim_data.map(filter_outliers)
        if hasattr(self.ensemble, "adjoints") and self.ensemble.adjoints is not None:
            self.ensemble.adjoints = self.ensemble.adjoints.map(filter_outliers)


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
            elif hasattr(self.scheme, save_type):
                save_attr = getattr(self.scheme, save_type)
                if isinstance(save_attr, (pd.DataFrame, PETDataFrame)):
                    save_dict[save_type] = save_attr.to_dict(orient='records')
                else:
                    save_dict[save_type] = save_attr
            elif save_type == "state":
                save_dict.update(self._state_debug_dict())
            else:
                print(f"Cannot save {save_type}, because it is a local variable!\n\n")

        save_dict["savefolder"] = self.save_folder
        at.save_analysisdebug(self.scheme.iteration, **save_dict)

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
        """Run forecast simulations and prepare predicted data for analysis.

        Retained as a thin delegation: the forecast itself now lives on the
        ensemble, as :meth:`pipt.ensembles.ForecastMixin.forecast`.
        """
        self.ensemble.forecast()
