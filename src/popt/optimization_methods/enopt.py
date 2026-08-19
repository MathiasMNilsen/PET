"""Ensemble optimization methods compatible with OptimizerBase."""

import numpy as np
import pprint
from scipy.optimize import OptimizeResult

from popt.misc_tools import optim_tools as ot
from popt.optimization_methods.optimizer_base import OptimizerBase
import popt.optimization_methods.subroutines.optimizers as opt

__author__ = ""
__all__ = ["EnOpt"]


class EnOpt(OptimizerBase):
    """Ensemble-based optimization (EnOpt)."""

    VALID_OPTIMIZERS = ("GD", "Adam", "AdaMax", "Steihaug")

    def __init__(self, fun, x, jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """Initialize an EnOpt optimizer instance.

        Parameters
        ----------
        fun : callable
            Objective function.
        x : ndarray
            Initial control/state vector.
        jac : callable
            Ensemble gradient function.
        hess : callable, optional
            Ensemble Hessian function.
        args : tuple, optional
            The first tuple element is interpreted as the initial covariance.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            EnOpt and OptimizerBase configuration.
            - maxiter: Maximum number of iterations (default: 100).
            - tol: Convergence tolerance for objective improvement (default: 1e-6).
            - ftol: Function tolerance used by the shared optimizer base. Defaults to ``tol`` when provided.
            - step_size: Initial optimizer step size. Overrides ``alpha`` when provided.
            - alpha: Initial optimizer step size (default: 0.1).
            - alpha_cov: Covariance update scaling factor (default: 0.001).
            - beta: Momentum parameter used in the optimizer and optional Nesterov updates (default: 0.0).
            - nesterov: Whether to evaluate search quantities with Nesterov momentum (default: False).
            - alpha_maxiter: Maximum number of backtracking trials per iteration (default: 5).
            - resample: Number of covariance resampling attempts if no improvement is found (default: 0).
            - hessian: Whether to use the Hessian in the search direction computation (default: False).
            - normalize: Whether to normalize the gradient or Hessian-derived search quantities (default: True).
            - cov_factor: Covariance shrink factor applied during resampling (default: 0.5).
            - optimizer: Update rule name. Supported values are ``GD``, ``Adam``, ``AdaMax``, and ``Steihaug`` (default: ``GD``).
            - gtol: Tolerance for convergence based on projected gradient infinity norm (default: 1e-5).
            - savefolder: Directory used when persisting iteration results (default: ``Iteration_Results``).
            - saveit: Whether to save optimization results at each iteration (default: False).
            - fun0: Initial objective value to reuse instead of recomputing it.
            - jac0: Initial gradient value to reuse instead of recomputing it.
            - hess0: Initial Hessian value to reuse instead of recomputing it.
            - restart: Restart optimization from a restart file (default: False).
            - restartsave: Save a restart file after each successful iteration (default: False).
            - restart_file: Restart file path.
            - logit: Enable optimizer logging.
            - logger_name: Log file name.
            - epf: Optional EPF settings handled by OptimizerBase.
        """
        if jac is None:
            raise ValueError("EnOpt requires a Jacobian (ensemble gradient) callable.")

        # Keep args empty for wrapped callables to avoid duplicating covariance
        # (EnOpt passes covariance explicitly during each update).
        super().__init__(x0=x, fun=fun, jac=jac, hess=hess, args=(), bounds=bounds, **options)

        self.callback = callback if callable(callback) else None

        # EnOpt controls
        self.obj_func_tol = options.get("tol", 1e-6)
        self.ftol = options.get("tol", options.get("ftol", self.ftol))
        self.alpha = options.get("step_size", options.get("alpha", 0.1))
        self.alpha_cov = options.get("alpha_cov", 0.001)
        self.beta = options.get("beta", 0.0)
        self.nesterov = options.get("nesterov", False)
        self.alpha_iter_max = options.get("alpha_maxiter", 5)
        self.max_resample = options.get("resample", 0)
        self.use_hessian = options.get("hessian", False)
        self.normalize = options.get("normalize", True)
        self.cov_factor = options.get("cov_factor", 0.5)
        self.gtol = options.get("gtol", 1e-5)
        self.savefolder = options.get("savefolder", "Iteration_Results")

        # Dynamic EnOpt state
        self.cov = np.asarray(args[0], dtype=float)
        self.state_step = np.zeros_like(self.xk, dtype=float)
        self.cov_step = np.zeros_like(self.cov, dtype=float)
        self.alpha_iter = 0

        self.optimizer_name = options.get("optimizer", "GD")
        self.optimizer = self._build_optimizer(self.optimizer_name)

        if self._maybe_restore_restart():
            # Backward compatibility alias used in legacy code.
            self.obj_func_values = self.fk
            return

        # Initial callable values
        self.fk = options.get("fun0", None)
        self.jk = options.get("jac0", None)
        self.hk = options.get("hess0", None)

        if self.fk is None:
            if self.logger:
                self.logger('Computing initial function value...')
            self.fk = self.fun(self.xk)

        self.obj_func_values = self.fk

        if self.logger:
            self.logger("========== Starting EnOpt Minimization ==========")
            if self.options:
                self.logger(f"\n\nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n")

        self._log_iteration()
        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

    @classmethod
    def minimize(cls, x0, fun, jac, hess=None, args=(), bounds=None, callback=None, **options):
        """Run EnOpt and return OptimizeResult.

        Parameters
        ----------
        x0 : ndarray
            Initial control/state vector.
        fun : callable
            Objective function.
        jac : callable
            Ensemble gradient function.
        hess : callable, optional
            Ensemble Hessian function.
        args : tuple, optional
            The first tuple element is interpreted as the initial covariance.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            EnOpt and OptimizerBase configuration.
            - maxiter: Maximum number of iterations (default: 100).
            - tol: Convergence tolerance for objective improvement (default: 1e-6).
            - ftol: Function tolerance used by the shared optimizer base. Defaults to ``tol`` when provided.
            - step_size: Initial optimizer step size. Overrides ``alpha`` when provided.
            - alpha: Initial optimizer step size (default: 0.1).
            - alpha_cov: Covariance update scaling factor (default: 0.001).
            - beta: Momentum parameter used in the optimizer and optional Nesterov updates (default: 0.0).
            - nesterov: Whether to evaluate search quantities with Nesterov momentum (default: False).
            - alpha_maxiter: Maximum number of backtracking trials per iteration (default: 5).
            - resample: Number of covariance resampling attempts if no improvement is found (default: 0).
            - hessian: Whether to use the Hessian in the search direction computation (default: False).
            - normalize: Whether to normalize the gradient or Hessian-derived search quantities (default: True).
            - cov_factor: Covariance shrink factor applied during resampling (default: 0.5).
            - optimizer: Update rule name. Supported values are ``GD``, ``Adam``, ``AdaMax``, and ``Steihaug`` (default: ``GD``).
            - gtol: Tolerance for convergence based on projected gradient infinity norm (default: 1e-5).
            - savefolder: Directory used when persisting iteration results (default: ``Iteration_Results``).
            - saveit: Whether to save optimization results at each iteration (default: False).
            - fun0: Initial objective value to reuse instead of recomputing it.
            - jac0: Initial gradient value to reuse instead of recomputing it.
            - hess0: Initial Hessian value to reuse instead of recomputing it.
            - restart: Restart optimization from a restart file (default: False).
            - restartsave: Save a restart file after each successful iteration (default: False).
            - restart_file: Restart file path.
            - logit: Enable optimizer logging.
            - logger_name: Log file name.
            - epf: Optional EPF settings handled by OptimizerBase.

        Returns
        -------
        OptimizeResult
            The optimization result.
        """
        optimizer = cls(
            fun=fun,
            x=x0,
            args=args,
            jac=jac,
            hess=hess,
            bounds=bounds,
            callback=callback,
            **options,
        )
        optimizer.run_optimization()
        return optimizer.optimize_results

    def update_step(self) -> bool:
        """Perform one EnOpt step with backtracking and optional resampling."""
        self.optimizer.restore_parameters()
        resampling_iter = 0

        while resampling_iter <= self.max_resample:
            shrink = self.cov_factor ** resampling_iter
            self._apply_optimizer_backtracking(np.sqrt(self.cov_factor) ** resampling_iter)
            self.jk, self.hk = self._compute_search_quantities(shrink)

            self.alpha_iter = 0
            while self.alpha_iter <= self.alpha_iter_max:
                new_state, new_step = self.optimizer.apply_update(
                    self.xk,
                    self.jk,
                    hessian=self.hk,
                    iter=self.iteration,
                )
                new_state = self.bound_handler.project_to_bounds(new_state)
                new_func_values = self.fun(new_state)

                if np.mean(self.fk) - np.mean(new_func_values) > self.obj_func_tol:
                    self._accept_step(new_state, new_func_values, new_step, self.hk)
                    return True

                if self.alpha_iter < self.alpha_iter_max:
                    self._apply_optimizer_backtracking()
                    self.alpha_iter += 1
                else:
                    break

            if (resampling_iter < self.max_resample) and (np.mean(new_func_values) > np.mean(self.fk)):
                resampling_iter += 1
                self.optimizer.restore_parameters()
                continue

            self.conv_msg = "EnOpt failed to find an improving step."
            return False

        self.conv_msg = "EnOpt exhausted all resampling attempts."
        return False

    def check_convergence(self) -> bool:
        """Check convergence using projected gradient infinity norm."""
        proj_jac = self.bound_handler.project_gradient(self.xk, self.jk)
        if np.linalg.norm(proj_jac, np.inf) < self.gtol:
            self.conv_msg = f"Projected gradient norm ‖g‖∞ < {self.gtol}."
            return True
        return False

    def _compute_search_quantities(self, shrink):
        cov_step = self.beta * self.cov_step if self.nesterov else 0.0
        state_step = self.beta * self.state_step if self.nesterov else 0.0

        cov = shrink * (self.cov + cov_step)
        x_for_grad = self.xk + state_step

        gradient = self.jac(x_for_grad, cov, epf=self.epf)
        hessian = self.hess(x_for_grad, cov) if self.hess is not None else None

        if self.use_hessian:
            gradient = np.linalg.inv(hessian) @ (self.cov @ self.cov) @ gradient
        elif self.normalize:
            gradient = gradient / np.maximum(np.linalg.norm(gradient, np.inf), 1e-12)

        if self.normalize and hessian is not None:
            hessian = hessian / np.maximum(np.linalg.norm(hessian, np.inf), 1e-12)

        return gradient, hessian

    def _accept_step(self, new_state, new_func_values, new_step, hessian):
        self.xk_old = self.xk
        self.fk_old = self.fk

        self.xk = new_state
        self.fk = new_func_values
        self.obj_func_values = self.fk
        self.state_step = new_step
        if hasattr(self.optimizer, "get_step_size"):
            self.alpha = self.optimizer.get_step_size()

        if hessian is not None:
            grad_cov = self.bound_handler.hess_from_unit_cube(hessian)
            self.cov_step = self.alpha_cov * grad_cov + self.beta * self.cov
            self.cov = ot.get_sym_pos_semidef(self.cov - self.cov_step)

        if self.xk.size == 1 and hasattr(self.optimizer, "step_size"):
            self.optimizer.step_size /= 2

        self.optimizer.restore_parameters()

        if callable(self.callback):
            self.callback(self)

        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

        self._log_iteration()

    def _build_optimizer(self, optimizer_name):
        if optimizer_name not in self.VALID_OPTIMIZERS:
            raise ValueError(
                f"Optimizer '{optimizer_name}' not recognized for EnOpt. "
                f"Valid options are: {self.VALID_OPTIMIZERS}."
            )

        if optimizer_name == "GD":
            return opt.GradientDescent(self.alpha, self.beta)
        if optimizer_name == "Adam":
            return opt.Adam(self.alpha, self.beta)
        if optimizer_name == "AdaMax":
            self.normalize = False
            return opt.AdaMax(self.alpha, self.beta)
        return opt.Steihaug(delta0=3.0)

    def _apply_optimizer_backtracking(self, shrink=0.5):
        try:
            self.optimizer.apply_backtracking(shrink)
        except TypeError:
            self.optimizer.apply_backtracking()

    def _get_restart_state(self) -> dict:
        return {
            "cov": self.cov,
            "state_step": self.state_step,
            "cov_step": self.cov_step,
            "alpha": self.alpha,
            "alpha_iter": self.alpha_iter,
            "obj_func_tol": self.obj_func_tol,
            "optimizer_name": self.optimizer_name,
            "optimizer_state": dict(self.optimizer.__dict__),
        }

    def _set_restart_state(self, state: dict) -> None:
        self.cov = state.get("cov", self.cov)
        self.state_step = state.get("state_step", self.state_step)
        self.cov_step = state.get("cov_step", self.cov_step)
        self.alpha = state.get("alpha", self.alpha)
        self.alpha_iter = state.get("alpha_iter", self.alpha_iter)
        self.obj_func_tol = state.get("obj_func_tol", self.obj_func_tol)

        self.optimizer_name = state.get("optimizer_name", self.optimizer_name)
        self.optimizer = self._build_optimizer(self.optimizer_name)
        self.optimizer.__dict__.update(state.get("optimizer_state", {}))
        self.obj_func_values = self.fk

    def _log_iteration(self) -> None:
        if self.logger:
            info = {
                "iter.": self.iteration,
                "alpha_iter": self.alpha_iter,
                "obj_func": float(np.mean(self.fk)),
                "step-size": self.alpha,
                "cov[0,0]": float(self.cov[0, 0]),
            }
            if self.epf:
                info["EPF iter."] = self.epf_iteration
            self.logger(**info)






