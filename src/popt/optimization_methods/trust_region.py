"""Trust-region deterministic optimization methods.

This module implements a trust-region optimizer with optional
BFGS Hessian approximation and restart support.
"""

import numpy as np
import pprint
from scipy.optimize import OptimizeResult

# Internal imports
from popt.misc_tools import optim_tools as ot
from popt.optimization_methods.optimizer_base import OptimizerBase
from popt.optimization_methods.subroutines.subroutines import solve_trust_region_subproblem

__author__ = "Mathias Methlie Nilsen"
__all__ = ["TrustRegion"]

# Symbols for logger output
subk = "\u2096"
fun_xk_symbol = f"fun(x{subk})"
delta_k_symbol = f"\u0394{subk}"
rho_symbol = f"\u03C1{subk}"
jac_inf_symbol = f"\u2016jac(x{subk})\u2016\u221E"


class TrustRegion(OptimizerBase):
    """Trust-region optimizer compatible with OptimizerBase.

    The class supports exact Hessian trust-region subproblems (iterative or
    CG-Steihaug) and optional BFGS Hessian approximation via ``hess='BFGS'``.
    """

    VALID_METHODS = ("iterative", "CG-Steihaug")

    def __init__(
        self,
        x0,
        fun,
        jac,
        hess,
        method="iterative",
        args=(),
        bounds=None,
        callback=None,
        **options,
    ):
        """Initialize a trust-region optimizer instance."""
        if jac is None:
            raise ValueError("TrustRegion requires a Jacobian (gradient) function.")

        use_bfgs = isinstance(hess, str) and hess.upper() == "BFGS"
        if (not use_bfgs) and (hess is None):
            raise ValueError("TrustRegion requires a Hessian function or hess='BFGS'.")

        super().__init__(x0, fun, jac, None if use_bfgs else hess, args, bounds, **options)

        self.callback = callback if callable(callback) else None
        self.method = self._validate_method(method)
        self.quasi_newton = use_bfgs

        convergence_criteria = options.get("convergence_criteria", None)
        self.convergence_criteria = convergence_criteria if callable(convergence_criteria) else None

        # Trust-region controls
        self.trust_radius = options.get("trust_radius", 1.0)
        self.trust_radius_max = options.get("trust_radius_max", 100 * self.trust_radius)
        self.trust_radius_min = options.get("trust_radius_min", self.trust_radius / 1000)
        self.trust_radius_cuts = options.get("trust_radius_cuts", 4)

        # Acceptance and radius updates
        self.rho_tol = options.get("rho_tol", 1e-6)
        self.eta1 = options.get("eta1", 0.05) # Threshold for rejecting a step
        self.eta2 = options.get("eta2", 0.5)  # Threshold for increasing the trust-region radius
        self.gam1 = options.get("gam1", 0.5)  # Factor to decrease the trust-region radius when a step is rejected
        self.gam2 = options.get("gam2", 1.5)  # Factor to increase the trust-region radius when a step is accepted and hits the boundary
        self.rho = 0.0

        # Other options
        self.resample = options.get("resample", False)
        self.gtol = options.get("gtol", 1e-5)
        self.savefolder = options.get("savefolder", "Iteration_Results")
        self.jk_old = None

        if self._maybe_restore_restart():
            return

        # Initial callable values
        self.fk = options.get("fun0", None)
        self.jk = options.get("jac0", None)
        self.hk = options.get("hess0", None)

        if self.fk is None:
            self.fk = self._objective_value(self.xk)
        if self.jk is None:
            self.jk = self.jac(self.xk)
        if self.hk is None and (not self.quasi_newton):
            self.hk = self.hess(self.xk)

        if self.logger:
            self.logger("========== Starting Trust-Region Minimization ==========")
            if self.options:
                self.logger(f"\n\nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n")

        self._log_iteration()

        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

    @classmethod
    def minimize(
        cls,
        x0,
        fun,
        jac,
        hess,
        method="iterative",
        args=(),
        bounds=None,
        callback=None,
        **options,
    ):
        """Run the optimization process and return results."""
        optimizer = cls(
            x0,
            fun,
            jac,
            hess,
            method=method,
            args=args,
            bounds=bounds,
            callback=callback,
            **options,
        )
        optimizer.optimization_loop()
        return optimizer.optimize_results

    def update_step(self) -> bool:
        """Perform one trust-region step with optional radius reductions."""
        if self.jk is None:
            self.jk = self.jac(self.xk)
        if self.hk is None and (not self.quasi_newton):
            self.hk = self.hess(self.xk)

        return self._attempt_step(inner_iter=0)

    def check_convergence(self) -> bool:
        """Check convergence via projected gradient infinity norm."""
        proj_jac = self.bound_handler.project_gradient(self.xk, self.jk)
        if np.linalg.norm(proj_jac, np.inf) < self.gtol:
            self.conv_msg = f"Projected gradient norm ‖g‖∞ < {self.gtol}."
            return True

        if self.trust_radius <= self.trust_radius_min:
            self.conv_msg = f"Trust-region radius {delta_k_symbol} <= {self.trust_radius_min}."
            return True

        if callable(self.convergence_criteria) and self.convergence_criteria(self):
            self.conv_msg = "Custom convergence criteria met."
            return True

        return False

    def _attempt_step(self, inner_iter: int) -> bool:
        if inner_iter > self.trust_radius_cuts:
            self.conv_msg = "Trust-region step rejected after radius cut attempts."
            return False

        jk_proj = self.bound_handler.project_gradient(self.xk, self.jk)

        if self.quasi_newton and (self.hk is None) and (self.iteration == 1):
            sk = -jk_proj
            sk_norm = np.linalg.norm(sk, np.inf)
            if sk_norm > 0:
                sk = sk / sk_norm * self.trust_radius
            hits_boundary = True
        else:
            hk_step = self.hk
            if hk_step is None:
                hk_step = self.hess(self.xk)
                self.hk = hk_step

            if callable(self.method):
                sk, hits_boundary = self.method(
                    self.xk,
                    self.fk,
                    jk_proj,
                    hk_step,
                    self.trust_radius,
                    **self.options,
                )
            else:
                sk, hits_boundary = solve_trust_region_subproblem(
                    self.xk,
                    self.fk,
                    jk_proj,
                    hk_step,
                    self.trust_radius,
                    method=self.method,
                    **self.options,
                )

        xk_new = self.bound_handler.project_to_bounds(self.xk + sk)
        fk_new = self._objective_value(xk_new)

        df = self.fk - fk_new
        if self.quasi_newton and (self.iteration == 1) and (self.hk is None):
            dm = -np.dot(jk_proj, sk)
        else:
            hk_for_dm = self.hk
            if hk_for_dm is None:
                hk_for_dm = self.hess(self.xk)
            dm = -np.dot(jk_proj, sk) - 0.5 * np.dot(sk, hk_for_dm @ sk)

        self.rho = df / dm if dm != 0 else -np.inf

        if (self.rho > self.rho_tol) and (fk_new < self.fk):
            self._accept_step(xk_new, fk_new, sk, jk_proj, hits_boundary)
            return True

        if self.logger:
            if not (fk_new < self.fk):
                self.logger(
                    f"Function value not reduced: {fun_xk_symbol} = {fk_new:<10.4e} >= {self.fk:<10.4e}"
                )
            else:
                self.logger(
                    f"Step not successful: {rho_symbol} = {self.rho:<10.4e} < {self.rho_tol:<10.4e}"
                )

        old_radius = self.trust_radius
        self.trust_radius *= 0.25
        if self.logger:
            self.logger(
                f"Reducing {delta_k_symbol}: {old_radius:<10.4e} -> {self.trust_radius:<10.4e}"
            )

        if self.trust_radius < self.trust_radius_min:
            self.conv_msg = f"Trust-region radius {delta_k_symbol} below minimum."
            return False

        if self.resample:
            self.jk = self.jac(self.xk)
            if not self.quasi_newton:
                self.hk = self.hess(self.xk)

        return self._attempt_step(inner_iter=inner_iter + 1)

    def _accept_step(self, xk_new, fk_new, sk, jk_proj, hits_boundary) -> None:
        self.xk_old = self.xk
        self.fk_old = self.fk
        self.jk_old = self.jk

        self.xk = xk_new
        self.fk = fk_new
        self.jk = self.jac(self.xk)

        if self.quasi_newton:
            yk = self.jk - self.jk_old
            if self.hk is None:
                denom = np.dot(yk, sk)
                if denom > 0:
                    self.hk = np.dot(yk, yk) / denom * np.eye(self.xk.size)
                else:
                    self.hk = np.eye(self.xk.size)
            self.hk = self._bfgs_update(self.hk, sk, yk)
        else:
            self.hk = self.hess(self.xk)

        self._update_trust_radius(hits_boundary)

        if callable(self.callback):
            self.callback(self)

        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

        self._log_iteration(hits_boundary=hits_boundary)

    def _update_trust_radius(self, hits_boundary: bool) -> None:
        delta_old = self.trust_radius

        if (self.rho >= self.eta2) and hits_boundary:
            delta_new = min(self.gam2 * delta_old, self.trust_radius_max)
        elif self.rho < self.eta1:
            delta_new = self.gam1 * delta_old
        else:
            delta_new = delta_old

        self.trust_radius = np.clip(delta_new, self.trust_radius_min, self.trust_radius_max)

        if self.logger and (self.trust_radius != delta_old):
            d_delta = (self.trust_radius - delta_old) / delta_old * 100
            self.logger(
                f"Tr-radius {delta_k_symbol} updated: {delta_old:<10.4e} -> {self.trust_radius:<10.4e} ({d_delta:<.2f}%)"
            )

    def _objective_value(self, x) -> float:
        return float(np.mean(self.fun(x)))

    def _validate_method(self, method):
        if callable(method):
            if self.logger:
                self.logger("Using custom trust-region subproblem solver callable.")
            return method

        if not isinstance(method, str):
            raise ValueError("Method must be a string or a callable.")

        if method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid trust-region method '{method}'. Valid options are: {self.VALID_METHODS}."
            )

        return method

    def _bfgs_update(self, Bk, sk, yk):
        sk = sk.reshape(-1, 1)
        yk = yk.reshape(-1, 1)

        ykTsk = (yk.T @ sk).item()
        skTBksk = (sk.T @ Bk @ sk).item()
        if ykTsk <= 0 or skTBksk <= 0:
            return Bk

        term1 = np.matmul(yk, yk.T) / ykTsk
        term2 = np.matmul(np.matmul(Bk, sk), np.matmul(sk.T, Bk)) / skTBksk
        return Bk + term1 - term2

    def _get_restart_state(self) -> dict:
        return {
            "trust_radius": self.trust_radius,
            "rho": self.rho,
            "jk_old": self.jk_old,
            "quasi_newton": self.quasi_newton,
        }

    def _set_restart_state(self, state: dict) -> None:
        self.trust_radius = state.get("trust_radius", self.trust_radius)
        self.rho = state.get("rho", self.rho)
        self.jk_old = state.get("jk_old", self.jk_old)
        self.quasi_newton = state.get("quasi_newton", self.quasi_newton)

    def _log_iteration(self, step_norm=None, hits_boundary=None) -> None:
        if self.logger:
            info = {
                "iter.": self.iteration,
                fun_xk_symbol: self.fk,
                delta_k_symbol: self.trust_radius,
                rho_symbol: self.rho,
                #jac_inf_symbol: np.linalg.norm(self.jk, np.inf),
            }
            if step_norm is not None:
                info[f"|p{subk}|∞"] = step_norm
            if hits_boundary is not None:
                info[f"\u2016p{subk}\u2016 = {delta_k_symbol}"] = "yes" if hits_boundary else "no"
            if self.epf:
                info["EPF iter."] = self.epf_iteration
            self.logger(**info)
