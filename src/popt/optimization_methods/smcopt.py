"""Stochastic Monte-Carlo optimization compatible with OptimizerBase."""

import numpy as np
import pprint
from scipy.optimize import OptimizeResult

from popt.misc_tools import optim_tools as ot
from popt.optimization_methods.optimizer_base import OptimizerBase
import popt.optimization_methods.subroutines.optimizers as opt

__author__ = ""
__all__ = ["SmcOpt"]


class SmcOpt(OptimizerBase):
    """Sequential Monte-Carlo optimizer with resampling and backtracking."""

    def __init__(self, fun, x, args=(), sens=None, bounds=None, callback=None, **options):
        """
        Parameters
        ----------
        fun : callable
            objective function

        x : ndarray
            Initial state

        args : tuple
            Initial covariance tuple where ``args[0]`` is the covariance matrix used for sampling.

        sens : callable
            Ensemble sensitivity function

        bounds : list, optional
            (min, max) pairs for each element in x. None is used to specify no bound.

        callback : callable, optional
            Callback invoked after successful updates.

        options : dict
            Optimization options

            - maxiter: maximum number of iterations (default 100)
            - restart: restart optimization from a restart file (default false)
            - restartsave: save a restart file after each successful iteration (default false)
            - restart_file: restart file path
            - tol: convergence tolerance for the objective function (default 1e-6)
            - alpha: weight between previous and new step (default 0.1)
            - alpha_maxiter: maximum number of backtracking trials (default 5)
            - resample: number indicating how many times resampling is tried if no improvement is found
            - cov_factor: factor used to shrink the covariance for each resampling trial (default 0.5)
            - inflation_factor: term used to weight down prior influence (default 1.0)
            - survival_factor: fraction of surviving samples (clipped to [0.1, 1.0])
            - logit: enable optimizer logging (default true)
            - logger_name: log file name (default OPTIM.log)
            - saveit: save intermediate optimize results (default false)
            - savefolder/save_folder: folder used when saveit is true
            - epf: optional EPF settings handled by OptimizerBase
        """
        if sens is None or not callable(sens):
            raise ValueError("SmcOpt requires a callable sensitivity function 'sens'.")
        if len(args) < 1:
            raise ValueError("SmcOpt requires initial covariance as args[0].")

        # SmcOpt historically operates in physical coordinates.
        options = {**options, "transform": False}
        super().__init__(x0=x, fun=fun, jac=None, hess=None, args=(), bounds=bounds, **options)

        self.callback = callback if callable(callback) else None
        self.sens = sens

        # SmcOpt controls
        self.obj_func_tol = options.get("tol", 1e-6)
        self.ftol = options.get("tol", options.get("ftol", self.ftol))
        self.alpha = options.get("alpha", 0.1)
        self.alpha_iter_max = options.get("alpha_maxiter", 5)
        self.max_resample = options.get("resample", 0)
        self.cov_factor = options.get("cov_factor", 0.5)
        self.inflation_factor = options.get("inflation_factor", 1.0)
        self.survival_factor = float(np.clip(options.get("survival_factor", 1.0), 0.1, 1.0))
        self.savefolder = options.get("savefolder", options.get("save_folder", "./"))
        self.alpha_iter = 0

        # Dynamic SMC state
        self.cov = np.asarray(args[0], dtype=float)
        self.best_state = None
        self.best_func = None
        self.sens_njev = 0

        self.optimizer = opt.GradientDescent(self.alpha, 0.0)

        if self._maybe_restore_restart():
            self.obj_func_values = self.fk
            return

        self.fk = options.get("fun0", None)
        self.jk = options.get("jac0", None)
        self.hk = options.get("hess0", None)

        if self.fk is None:
            self.fk = self.fun(self.xk)

        self.obj_func_values = self.fk
        self.best_func = float(np.mean(options.get("best_func", self.fk)))

        if self.logger:
            self.logger("========== Starting SmcOpt Minimization ==========")
            if self.options:
                self.logger(f"\n\nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n")

        self._log_iteration()
        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

        if options.get("autorun", True):
            self.optimization_loop()
            self.optimize_results = self._update_optimize_result()

    @classmethod
    def minimize(cls, x0, fun, sens, args=(), bounds=None, callback=None, **options):
        """Run SmcOpt and return OptimizeResult."""
        optimizer = cls(
            fun=fun,
            x=x0,
            args=args,
            sens=sens,
            bounds=bounds,
            callback=callback,
            **{**options, "autorun": False},
        )
        optimizer.optimization_loop()
        return optimizer.optimize_results

    def update_step(self) -> bool:
        """Perform one SMC update step with backtracking and optional resampling."""
        self.optimizer.restore_parameters()
        resampling_iter = 0
        inflate = 2.0 * (self.inflation_factor + self.iteration)

        while resampling_iter <= self.max_resample:
            shrink = self.cov_factor ** resampling_iter
            self.optimizer.apply_backtracking(np.sqrt(self.cov_factor) ** resampling_iter)

            sens_matrix, self.best_state, best_func_tmp = self.sens(
                self.xk,
                inflate,
                shrink * self.cov,
                self.survival_factor,
                epf=self.epf,
            )
            self.sens_njev += 1

            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state = self.optimizer.apply_smc_update(self.xk, sens_matrix, iter=self.iteration)
                new_state = self.bound_handler.project_to_bounds(new_state)

                new_func_values = self.fun(new_state)

                improved_objective = np.mean(self.fk) - np.mean(new_func_values) > self.obj_func_tol
                improved_best = (self.best_func - best_func_tmp) > self.obj_func_tol
                if improved_objective or improved_best:
                    self._accept_step(new_state, new_func_values, best_func_tmp, improved_best)
                    return True

                if self.alpha_iter < self.alpha_iter_max:
                    self.optimizer.apply_backtracking()
                    self.alpha_iter += 1
                else:
                    break

            if (resampling_iter < self.max_resample) and (np.mean(new_func_values) > np.mean(self.fk)):
                resampling_iter += 1
                self.optimizer.restore_parameters()
                continue

            self.conv_msg = "SmcOpt failed to find an improving step."
            return False

        self.conv_msg = "SmcOpt exhausted all resampling attempts."
        return False

    def check_convergence(self) -> bool:
        # SmcOpt relies on shared function/state convergence checks in OptimizerBase.
        return False

    def _accept_step(self, new_state, new_func_values, best_func_tmp, improved_best):
        self.xk_old = self.xk
        self.fk_old = self.fk

        self.xk = new_state
        self.fk = new_func_values
        self.obj_func_values = self.fk
        if improved_best:
            self.best_func = float(best_func_tmp)

        self.optimizer.restore_parameters()

        if callable(self.callback):
            self.callback(self)

        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

        self._log_iteration()

    def _update_optimize_result(self):
        result = super()._update_optimize_result()
        result["fun"] = float(np.mean(self.fk))
        result["njev"] = self.sens_njev
        result["best_func"] = self.best_func
        return result

    def _get_restart_state(self) -> dict:
        return {
            "cov": self.cov,
            "best_state": self.best_state,
            "best_func": self.best_func,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "obj_func_tol": self.obj_func_tol,
            "sens_njev": self.sens_njev,
            "optimizer_state": dict(self.optimizer.__dict__),
        }

    def _set_restart_state(self, state: dict) -> None:
        self.cov = state.get("cov", self.cov)
        self.best_state = state.get("best_state", self.best_state)
        self.best_func = state.get("best_func", self.best_func)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        self.obj_func_tol = state.get("obj_func_tol", self.obj_func_tol)
        self.sens_njev = state.get("sens_njev", self.sens_njev)
        self.optimizer.__dict__.update(state.get("optimizer_state", {}))
        self.obj_func_values = self.fk

    def _log_iteration(self) -> None:
        if self.logger:
            info = {
                "iter.": self.iteration,
                "alpha_iter": self.alpha_iter,
                "obj_func": float(np.mean(self.fk)),
                "best_func": float(self.best_func),
                "step-size": self.alpha,
            }
            if self.epf:
                info["EPF iter."] = self.epf_iteration
            self.logger(**info)
