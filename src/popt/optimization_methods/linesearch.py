"""Line-search-based deterministic optimization methods.

This module implements gradient-based algorithms that share a common line
search interface, including gradient descent, BFGS, and Newton-CG.
"""

import numpy as np
import pprint
from scipy.optimize import OptimizeResult

# Internal imports
import popt.misc_tools.optim_tools as ot
from popt.optimization_methods.subroutines import line_search, line_search_backtracking, bfgs_update, newton_cg
from popt.optimization_methods.optimizer_base import OptimizerBase

__author__ = "Mathias Methlie Nilsen"
__all__ = ["LineSearch"]

# -----------------------------------------
# Some symbols for logger
# -----------------------------------------
subk = '\u2096'
sup2 = '\u00b2'
jac_inf_symbol = f'‖jac(x{subk})‖\u221E'
fun_xk_symbol  = f'fun(x{subk})'
nabla_symbol = "\u2207"


class LineSearch(OptimizerBase):
    """Line-search optimizer compatible with OptimizerBase.

    The class supports gradient descent, BFGS, and Newton-CG search
    directions, together with either Wolfe or backtracking line search.
    It can operate with bounds, optional state transformations, logging,
    result persistence, and restart checkpoints.
    """

    VALID_METHODS = ("GD", "BFGS", "Newton-CG")
    LS_METHODS = {
        0: line_search_backtracking, # Backtracking line search
        1: line_search,              # Wolfe line search
    }


    def __init__(self, x0, fun, method='GD', jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """Initialize a line-search optimizer instance.

        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector.
        fun : callable
            Objective function.
        method : {'GD', 'BFGS', 'Newton-CG'}, optional
            Search-direction method.
        jac : callable
            Gradient function.
        hess : callable, optional
            Hessian function, required by ``Newton-CG``.
        args : tuple, optional
            Extra positional arguments passed to the wrapped callables.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            Line-search and optimizer configuration.
            - step_size: Initial step size (default: None, auto-scaled).
            - step_size_max: Maximum step size (default: 1e5).
            - step_size_adapt: Step size adaptation strategy (0: none, 1: function-based, 2: gradient-based). Default is 1 (function-based).
            - c1: Armijo condition constant (default: 1e-4).
            - c2: Curvature condition constant (default: 0.9).
            - rho: Step size reduction factor for backtracking (default: 0.5).
            - lsmaxiter: Maximum line search iterations (default: 10).
            - lsmethod: Line search method (0: backtracking, 1: Wolfe, default: 1).
            - normalize: Whether to normalize the search direction (default: False).
            - recompute_jac: Number of gradient recomputation attempts on line search failure (default: 0).
            - saveit: Whether to save optimization results at each iteration (default: True).
            - gtol: Tolerance for convergence based on projected gradient infinity norm (default: 1e-5).
        """

        if jac is None:
            raise ValueError("LineSearch requires a Jacobian (gradient) function for the specified methods.")

        # Initialize the base class
        super().__init__(x0, fun, jac, hess, args, bounds, **options)

        # Validate method and required callables
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Valid options are: {self.VALID_METHODS}")
        if method in ("BFGS", "Newton-CG") and jac is None:
            raise ValueError(f"Method '{method}' requires a Jacobian (gradient) function.")
        if method == "Newton-CG" and hess is None:
            raise ValueError(f"Method '{method}' requires a Hessian function.")

        # Check for Callback function
        if callable(callback):
            self.callback = callback
        else:
            self.callback = None

        # Line search specific attributes
        self.method = method

        # Set options for step-size
        self.step_size       = options.get('step_size', None)
        self.step_size_max   = options.get('step_size_max', 1e5)
        self.step_size_adapt = options.get('step_size_adapt', 1)

        # Line search specific options
        self.line_search_options = {
            'c1': options.get('c1', 1e-4),              # Armijo condition constant
            'c2': options.get('c2', 0.9),               # Curvature condition constant
            'rho': options.get('rho', 0.5),             # Step size reduction factor for backtracking
            'amax': self.step_size_max,                 # Max step size for line search
            'lsmaxiter': options.get('lsmaxiter', 10),  # Max line search iterations
            'logger': self.logger,                      # Logger instance

        }
        try:
            lsmethod = options.get('lsmethod', 1)
            self.line_search_fn = self.LS_METHODS[lsmethod]
        except KeyError:
            raise ValueError(f"Invalid line search method: {lsmethod}")

        # Other options
        self.recompute_jac = options.get('recompute_jac', 0)
        self.normalize = options.get('normalize', False)
        self.savefolder = options.get('savefolder', 'Iteration_Results')
        self.saveit = options.get('saveit', False)
        self.jk_old = None
        self.pk_old = None
        self.gtol = options.get('gtol', 1e-5) # tolerance for inf-norm of jacobian

        if self.method == 'BFGS':
            self.bk = options.get('hess0_inv', np.eye(self.xk.size))  # BFGS approximation of the inverse Hessian

        if self._maybe_restore_restart():
            return

        # Check for initial callable values
        self.fk = options.get('fun0', None)
        self.jk = options.get('jac0', None)
        self.hk = options.get('hess0', None)

        # Initial logger message
        if self.logger:
            self.logger(f'========== Starting Line Search Minimization ({method}) ==========')
            if self.options:
                self.logger(f'\n \nUSER-SPECIFIED OPTIONS:\n{pprint.pformat(OptimizeResult(self.options))}\n')

        # Initial function and jacobian evaluation if not provided
        if self.fk is None:
            if self.logger:
                self.logger('Computing initial function value...')
            self.fk = self.fun(self.xk)
        if self.jk is None:
            if self.logger:
                self.logger('Computing initial jacobian...')
            self.jk = self.jac(self.xk)
        if self.hk is None and (self.hess is not None):
            if self.logger:
                self.logger('Computing initial Hessian...')
            self.hk = self.hess(self.xk)

        # Log initial values
        self._log_iteration()

        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

    @classmethod
    def minimize(cls, x0, fun, method='GD', jac=None, hess=None, args=(), bounds=None, callback=None, **options):
        """
        Run Line Search optimization.

        Parameters
        ----------
        x0 : ndarray
            Initial parameter vector.
        fun : callable
            Objective function.
        method : {'GD', 'BFGS', 'Newton-CG'}, optional
            Search-direction method. Default is 'GD' (Gradient Descent).
        jac : callable
            Gradient function.
        hess : callable, optional
            Hessian function, required by ``Newton-CG``.
        args : tuple, optional
            Extra positional arguments passed to the wrapped callables.
        bounds : sequence, optional
            Lower and upper bounds for each state variable.
        callback : callable, optional
            Callback invoked after successful updates.
        **options
            Line-search and optimizer configuration.
            - step_size: Initial step size (default: None, auto-scaled).
            - step_size_max: Maximum step size (default: 1e5).
            - step_size_adapt: Step size adaptation strategy (0: none, 1: function-based (default), 2: gradient-based).
            - c1: Armijo condition constant (default: 1e-4).
            - c2: Curvature condition constant (default: 0.9).
            - rho: Step size reduction factor for backtracking (default: 0.5).
            - lsmaxiter: Maximum line search iterations (default: 10).
            - lsmethod: Line search method (0: backtracking, 1: Wolfe, default: 1).
            - normalize: Whether to normalize the search direction (default: False).
            - recompute_jac: Number of gradient recomputation attempts on line search failure (default: 0).
            - saveit: Whether to save optimization results at each iteration (default: True).
            - gtol: Tolerance for convergence based on projected gradient infinity norm (default: 1e-5).

        Returns
        -------
        OptimizeResult
            The optimization result represented as a `scipy.optimize.OptimizeResult` object.
            - `x`: The solution array.
            - `fun`: The final objective function value.
            - `jac`: The final Jacobian (gradient) value.
            - `nfev`: The number of function evaluations.
            - `njev`: The number of Jacobian evaluations.
            - `message`: Description of the cause of termination.

        """
        optimizer = cls(
            x0,
            fun,
            method=method,
            jac=jac,
            hess=hess,
            args=args,
            bounds=bounds,
            callback=callback,
            **options
        )
        optimizer.optimization_loop()
        return optimizer.optimize_results


    def update_step(self) -> bool:
        """
        Perform one optimization step.

        The method computes a search direction, performs a line search, and
        updates optimizer state on success. When enabled, it can recompute the
        gradient and retry if the line search fails.

        Returns
        -------
        bool
            ``True`` if a valid step was accepted, otherwise ``False``.
        """
        iter_jac_recompute = 0  # Reset recompute counter for this step

        # Compute initial function and jacobian if not already available
        if self.jk is None:
            self.jk = self.jac(self.xk)
        if self.hk is None and (self.hess is not None):
            self.hk = self.hess(self.xk)

        # Perform line-search step (with optional recompute loop)
        while iter_jac_recompute <= self.recompute_jac:
            pk = self._compute_search_direction()
            step_size, fk_new, jk_new = self._run_line_search(pk)

            # SUCCESS --> accept step and return
            if step_size:
                self._accept_step(pk, step_size, fk_new, jk_new)
                return True

            # FAILURE --> recompute or exit
            self.conv_msg = 'Line search failed to find a suitable step size'
            if iter_jac_recompute < self.recompute_jac:
                if self.logger:
                    self.logger('Recomputing gradient and retrying line search...')
                self.jk = None
                iter_jac_recompute += 1
            else:
                return False


    def check_convergence(self) -> bool:
        """Check convergence using the projected infinity norm of the gradient."""
        # Check for convergence based on gradient norm
        proj_jac = self.bound_handler.project_gradient(self.xk, self.jk)
        if np.linalg.norm(proj_jac, np.inf) < self.gtol:
            self.conv_msg = f'Projected gradient norm ‖g‖∞ < {self.gtol}.'
            return True
        return False

    def _run_line_search(self, pk) -> tuple[float, float, np.ndarray]:
        """Run the line search algorithm to find an acceptable step size."""
        step_size = self._set_step_size(pk, self.step_size_max)
        step_size, fk_new, jk_new, _, _ = self.line_search_fn(
            step_size=step_size,
            xk=self.xk,
            pk=pk,
            fun=lambda x, *a, **kw: np.mean(self.fun(x, *a, **kw)),
            jac=self.jac,
            fk=np.mean(self.fk),
            jk=self.jk,
            **self.line_search_options
        )

        return step_size, fk_new, jk_new

    def _accept_step(self, pk, step_size, fk_new, jk_new) -> None:
        """Accept the proposed step and update the optimizer state."""
        self.xk_old = self.xk
        self.fk_old = self.fk
        self.jk_old = self.jk
        self.pk_old = pk

        self.xk = self.bound_handler.project_to_bounds(self.xk + step_size * pk)
        self.fk = fk_new
        self.jk = jk_new

        if callable(self.callback):
            self.callback(self)

        if self.method == 'BFGS':
            sk = self.xk - self.xk_old
            yk = self.jk - self.jk_old
            if self.iteration == 1:
                self.bk = np.dot(yk,sk)/np.dot(yk,yk) * np.eye(sk.size)
            self.bk = bfgs_update(self.bk, sk, yk)

        # Save Results
        self.optimize_results = self._update_optimize_result()
        if self.saveit:
            ot.save_optimize_results(self.optimize_results, folder=self.savefolder)

        # Invalidate Hessian for next iteration (will be recomputed if needed)
        self.hk = None

        # Log iteration results
        self._log_iteration(step_size=step_size)

    def _get_restart_state(self) -> dict:
        state = {
            'step_size': self.step_size,
            'jk_old': self.jk_old,
            'pk_old': self.pk_old,
        }
        if self.method == 'BFGS':
            state['bk'] = self.bk
        return state

    def _set_restart_state(self, state: dict) -> None:
        self.step_size = state.get('step_size', self.step_size)
        self.jk_old = state.get('jk_old', self.jk_old)
        self.pk_old = state.get('pk_old', self.pk_old)
        if self.method == 'BFGS' and 'bk' in state:
            self.bk = state['bk']


    def _compute_search_direction(self) -> np.ndarray:
        if self.method == 'GD':
            return -self.jk
        elif self.method == 'BFGS':
            return - np.matmul(self.bk, self.jk)
        elif self.method == 'Newton-CG':
            return newton_cg(self.jk, self.hk)
        else:
            raise ValueError(f"Unsupported method: {self.method}")


    def _set_step_size(self, pk, amax) -> float:
        if self.step_size is None:
            self.step_size = 0.25 / np.linalg.norm(pk, np.inf)

        alpha = float(np.asarray(self.step_size).reshape(-1)[0])

        if self.iteration > 1:
            slope = np.dot(pk, self.jk)
            if self.step_size_adapt == 1 and slope != 0:
                fk = float(np.asarray(np.mean(self.fk)).reshape(-1)[0])
                fk_old = float(np.asarray(np.mean(self.fk_old)).reshape(-1)[0])
                alpha = 2 * (fk - fk_old) / slope
            elif self.step_size_adapt == 2 and slope != 0:
                slope_old = np.dot(self.pk_old, self.jk_old)
                alpha = self.step_size * slope_old / slope

        alpha = float(abs(alpha))

        if alpha >= amax:
            alpha = 0.75 * amax

        return alpha

    def _log_iteration(self, step_size=None) -> None:
        """Log the current iteration summary."""
        if self.logger:
            info = {
                'iter.': self.iteration,
                fun_xk_symbol: self.fk,
                jac_inf_symbol: np.linalg.norm(self.jk, np.inf),
                'step-size': step_size if step_size is not None else self.step_size
            }
            if self.epf:
                info['EPF iter.'] = self.epf_iteration
            self.logger(**info)



















