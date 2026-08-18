"""Workflow that surrounds an assimilation run.

Diagnostics, artifact saving and outlier handling are not part of any
assimilation algorithm, but every PIPT run wants them. They used to live on
``pipt.loop.assimilation.Assimilate`` together with the iteration loop; when
the schemes took ownership of their own loop the loop went away and this
stayed, as a mixin the schemes compose with.

It is expressed entirely through the hooks
:class:`~pipt.update_schemes.core.AssimilationSchemeBase` calls, so the
base loop remains algorithm-only and a scheme that wants none of this simply
does not mix it in.

Hook order over a run::

    prior forecast
    after_forecast()                replace outliers in the prior
    score_prior()                   prior misfit (the scheme's, not this mixin's)
    after_prior_forecast()          QA on the prior, save prior artifacts
    for each iteration:
        calc_analysis()
        after_analysis()            refresh screened QAQC variance
        forecast
        after_forecast()            replace outliers
        score_and_commit()
        after_accepted_iteration()  iteration artifacts, QA/QC, restart
    after_loop()                    posterior, stop reason, summary
"""

import os
import pickle
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from misc.structures import PETDataFrame
import pipt.misc_tools.analysis_tools as at
import pipt.misc_tools.extract_tools as extract
from pipt.misc_tools.qaqc_tools import QAQC

__all__ = ["AssimilationWorkflowMixin"]


class AssimilationWorkflowMixin:
    """Diagnostics, saving and outlier handling around an assimilation run."""

    PRIOR_FORECAST_FILE = "prior_forecast.pkl"
    POSTERIOR_STATE_FILE = "posterior_state_estimate.npz"
    POSTERIOR_FORECAST_FILE = "posterior_forecast.pkl"
    STOP_REASON_FILE = "why_iter_loop_stopped.pkl"

    qaqc: QAQC | None = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def after_prior_forecast(self) -> None:
        """Handle the prior forecast: prior QA, saved artifacts.

        Outlier replacement is not done here: ``run_prior_forecast`` now routes
        the prior through :meth:`after_forecast` like every other forecast, so
        it has already happened by the time this runs -- and before
        :meth:`~pipt.update_schemes.core.AssimilationSchemeBase.score_prior`
        computes the misfit, which is the order the previous duplicate call
        produced.
        """
        self.qaqc = self._build_qaqc()

        self._run_prior_quality_assurance()
        self._save_prior_forecast()
        if "analysisdebug" in self.keys_da:
            self._save_analysis_debug()
        if "iterinfo" in self.keys_da:
            self._save_iteration_information()
        self._save_restart_snapshot()

    def after_analysis(self) -> None:
        """Between analysis and forecast: refresh screened QAQC variance."""
        self._refresh_screened_qaqc_datavar()

    def after_forecast(self) -> None:
        """Between forecast and scoring: replace outlier members.

        Ordering matters -- outliers are replaced before the misfit is scored,
        so the replacement feeds into the number the scheme sees.
        """
        if "remove_outliers" in self.keys_da:
            self.ensemble.remove_outliers()

    def after_accepted_iteration(self) -> None:
        """Persist iteration artifacts and run QA/QC after an accepted update."""
        if "iterinfo" in self.keys_da:
            self._save_iteration_information()
        if "analysisdebug" in self.keys_da:
            self._save_analysis_debug()

        if self.qaqc is not None:
            if "qc" in self.keys_da:
                self._set_qaqc()
                self.qaqc.calc_da_stat()
            if "qa" in self.keys_da:
                self._set_qaqc()
                self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
                self.qaqc.calc_kg()

        self._save_restart_snapshot()

    def after_loop(self, converged: bool) -> None:
        """Save the posterior and the reason the run stopped."""
        if self._saving_enabled:
            self._save_posterior_results()
            self._save_stop_reason(converged)
        self._log_convergence_summary()

    # ------------------------------------------------------------------
    # QA/QC
    # ------------------------------------------------------------------
    def _build_qaqc(self) -> QAQC | None:
        """Create QA/QC helper only when requested by the configuration."""
        qaqc_requested = (
            "qa" in self.keys_da
            or "qa" in self.sim.input_dict
            or "qc" in self.keys_da
        )
        if not qaqc_requested:
            return None

        return QAQC(
            self.keys_da | self.sim.input_dict,
            self.ensemble.obs_data,
            self.ensemble.datavar,
            self.logger,
            self.prior_info,
            self.sim,
            self.prior_enX.to_dict(),
        )

    def _set_qaqc(self) -> None:
        self.qaqc.set(self.pred_data, self.enX.to_dict(), self.lam)

    def _run_prior_quality_assurance(self) -> None:
        if self.qaqc is None or "qa" not in self.keys_da:
            return

        self._set_qaqc()
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))
        self.qaqc.calc_coverage()
        self.qaqc.calc_kg({"plot_all_kg": True, "only_log": False, "num_store": 5})

    def _refresh_screened_qaqc_datavar(self) -> None:
        """Update QAQC data variance after first-iteration data screening."""
        if self.qaqc is None:
            return
        if "qa" not in self.keys_da:
            return
        if not extract.is_enabled(self.keys_da.get("screendata", False)):
            return
        if self.iteration != 1:
            return

        self.logger.info("Recomputing Mahalanobis distance with updated datavar")
        self.qaqc.datavar = self.ensemble.datavar
        self.qaqc.calc_mahalanobis((1, "time", 2, "time", 1, None, 2, None))

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_restart_snapshot(self) -> None:
        if extract.is_enabled(self.keys_da.get("restartsave", False)):
            self.ensemble.save()

    def _save_prior_forecast(self) -> None:
        if not self._saving_enabled:
            return
        try:
            self.sim_data.to_pickle(self._save_path(self.PRIOR_FORECAST_FILE))
        except Exception:
            np.savez(self._save_path(self.PRIOR_FORECAST_FILE), sim_data=self.sim_data)

    def _save_posterior_results(self) -> None:
        """Save posterior state and forecast, falling back to pickle if needed."""
        try:
            np.savez(self._save_path(self.POSTERIOR_STATE_FILE), **self.enX.to_dict())
            self.sim_data.to_pickle(self._save_path(self.POSTERIOR_FORECAST_FILE))
        except Exception:
            with open(self._save_path(self.POSTERIOR_STATE_FILE), "wb") as file:
                pickle.dump(self.enX.to_dict(), file)
            with open(self._save_path(self.POSTERIOR_FORECAST_FILE), "wb") as file:
                pickle.dump(self.sim_data, file)

    def _save_stop_reason(self, converged: bool) -> None:
        if converged:
            reason = "Convergence criteria met. Stopping assimilation loop."
        else:
            reason = "Maximum iterations reached without convergence."
        self.logger.info(reason)

        why = self.why_stop.copy() if isinstance(self.why_stop, dict) else self.why_stop
        if why is not None:
            why["conv_string"] = reason

        with open(self._save_path(self.STOP_REASON_FILE), "wb") as file:
            pickle.dump(why, file, protocol=4)

    def _log_convergence_summary(self) -> None:
        if self.prev_data_misfit is None:
            return

        out_str = "\n Convergence was met."
        if self.prior_data_misfit > self.data_misfit:
            out_str += (
                f" Obj. function reduced from {self.prior_data_misfit:0.1f} "
                f"to {self.data_misfit:0.1f}"
            )
        self.logger(out_str)

    def _save_iteration_information(self) -> None:
        """Run configured iteration-info hooks."""
        for element in self._as_list(self.keys_da["iterinfo"]):
            if ".py" not in element:
                continue

            module_name = element.removesuffix(".py")
            iter_info_func = import_module(module_name)
            iter_info_func.main(self)

    def _save_analysis_debug(self) -> None:
        """Save the scheme attributes named by ``analysisdebug``.

        One file per iteration, ``debug_analysis_step_{iteration}.npz``, with
        iteration 0 describing the prior. ``state`` is special-cased: it
        expands to one array per state variable rather than a single entry.

        A name the scheme does not carry is reported and skipped rather than
        failing the run, since a variable can legitimately be absent for a
        given scheme -- ``lam`` exists for the Levenberg-Marquardt family and
        not for ES-MDA.
        """
        save_dict: dict[str, Any] = {}

        for save_type in self._as_list(self.keys_da["analysisdebug"]):
            if hasattr(self, save_type):
                save_attr = getattr(self, save_type)
                if isinstance(save_attr, (pd.DataFrame, PETDataFrame)):
                    save_dict[save_type] = save_attr.to_dict(orient="records")
                else:
                    save_dict[save_type] = save_attr
            elif save_type == "state":
                save_dict.update(self._state_debug_dict())
            else:
                print(
                    f"Cannot save '{save_type}' at iteration {self.iteration}: "
                    f"neither {type(self).__name__} nor its ensemble has an "
                    f"attribute by that name.\n"
                )

        save_dict["savefolder"] = self.save_folder
        at.save_analysisdebug(self.iteration, **save_dict)

    def _state_debug_dict(self) -> dict[str, Any]:
        if getattr(self.ensemble, "multilevel", None) is not None:
            return {
                f"state_level{level}": self.enX[level].to_dict()
                for level in range(self.ensemble.tot_level)
            }
        return self.enX.to_dict()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _save_path(self, filename: str) -> str:
        if self.save_folder is None:
            raise RuntimeError("Cannot save results because saving is disabled.")
        return os.path.join(self.save_folder, filename)
