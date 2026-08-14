"""Shared base class for iterative ensemble data-assimilation schemes.

This is the PIPT counterpart to
:mod:`popt.optimization_methods.optimizer_base`, and deliberately mirrors its
shape: the scheme object owns its own iteration loop, its convergence checks,
and its checkpoint/restart handling, while subclasses supply only the
algorithm-specific analysis step.

The two packages differ in what the iteration acts on. An optimizer is handed
callables (``fun``, ``jac``, ``hess``) and drives a control vector. An
assimilation scheme is handed an *ensemble* collaborator, which owns the state
realisations, the observed data, and the forward simulator.

Ensemble collaborator protocol
------------------------------
The scheme only relies on the following members, so anything satisfying them
can be substituted (a lightweight fake is used in the unit tests):

``ensemble.forecast()``
    Run the forward simulator on the current state and refresh ``pred_data``.
``ensemble.enX``
    State ensemble matrix, shape ``(nx, ne)``.
``ensemble.pred_data``
    Predicted data for the current state.
``ensemble.logger``
    A :class:`ensemble.logger.PetLogger`, or ``None``.

Relationship to the legacy design
---------------------------------
Historically a PIPT scheme *inherited* from ``pipt.loop.ensemble.Ensemble`` and
an external ``pipt.loop.assimilation.Assimilate`` object drove the loop. That
made every scheme simultaneously an algorithm and a data container, and made
the analysis flavour (``approx``/``full``/``subspace``) part of the class name.
Here the ensemble is a *collaborator* rather than a superclass, matching how
``OptimizerBase`` composes with its callables.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import OptimizeResult

from ensemble.checkpoint import RestartMixin
from ensemble.logger import PetLogger

__all__ = ["AssimilationSchemeBase", "AssimilationResult"]


class AssimilationResult(OptimizeResult):
    """Result of an assimilation run.

    A ``dict`` subclass with attribute access, mirroring
    :class:`scipy.optimize.OptimizeResult` so that PIPT and POPT results can be
    handled the same way. Typical fields:

    ``nit``
        Number of accepted iterations.
    ``success``
        Whether the run stopped on a convergence criterion rather than by
        exhausting ``maxiter``.
    ``message``
        Human-readable reason the run stopped.
    ``why_stop``
        Mapping of criterion name to whether it fired.
    ``data_misfit`` / ``prior_data_misfit``
        Final and initial mean data misfit.
    """


class AssimilationSchemeBase(RestartMixin, ABC):
    """Base class for iterative ensemble data-assimilation schemes.

    Subclasses implement :meth:`update_step`, which performs one analysis and
    reports whether the resulting step was accepted. Everything shared between
    schemes -- the loop, convergence bookkeeping, restart files, logging and
    the result object -- lives here.
    """

    def __init__(self, ensemble, **options):
        """
        Parameters
        ----------
        ensemble : object
            Collaborator satisfying the ensemble protocol described in the
            module docstring. Owns the state, the observed data and the
            forward simulator.
        **options
            Scheme configuration.

            - maxiter: Maximum number of accepted iterations (default: 100).
            - misfit_tol: Relative data-misfit tolerance for convergence
              (default: 0.01). The assimilation counterpart of an optimizer's
              ``ftol``.
            - step_tol: Absolute tolerance on the norm of the state update
              (default: 1e-8). Counterpart of an optimizer's ``xtol``.
            - logit: Enable logging (default: True).
            - logger_name: Log file name (default: 'ASSIM.log').
            - restart: Restore from a restart file on startup (default: False).
            - restartsave: Write a restart file after each accepted iteration
              (default: False).
            - restart_file: Path for the restart file
              (default: '{scheme_name}_restart.pkl').
        """
        self.ensemble = ensemble
        self.options = options

        # Core iteration controls.
        self.iteration = 0
        self.maxiter = options.get("maxiter", 100)

        # Convergence tolerances.
        self.misfit_tol = options.get("misfit_tol", 0.01)
        self.step_tol = options.get("step_tol", 1e-8)

        # Restart/checkpoint controls.
        self.restart = options.get("restart", False)
        self.restartsave = options.get("restartsave", False)
        self.restart_file = options.get(
            "restart_file",
            options.get("restartfile", f"{type(self).__name__.lower()}_restart.pkl"),
        )
        self._restart_loaded = False

        # Iteration state. `data_misfit` is the assimilation analogue of an
        # optimizer's objective value; `enX` of its control vector.
        self.data_misfit = None
        self.prior_data_misfit = None
        self.data_misfit_std = None
        self.prev_data_misfit = None
        self.enX_old = None

        # Logging.
        self.logger = None
        if options.get("logit", True):
            self.logger = PetLogger(options.get("logger_name", "ASSIM.log"))

        # Result container and stop bookkeeping.
        self.conv_msg = ""
        self.why_stop = {}
        self.results = AssimilationResult()

    # ------------------------------------------------------------------
    # Ensemble delegation
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        """Fall back to the ensemble for attributes the scheme does not own.

        The analysis strategies in :mod:`pipt.update_schemes.update_methods_ns`
        read their context off ``self`` -- ``keys_da``, ``proj``, ``cov_data``,
        ``localization`` and friends -- which resolved by inheritance while a
        scheme *was* an ensemble. Under composition they would not, so reads
        fall through to the collaborator instead. Replacing this with an
        explicit strategy context is the follow-on step noted in
        ``pipt/update_schemes/analysis/base.py``.

        Reads only. Assignments still land on the scheme, so anything the
        ensemble must actually see -- ``enX``, ``enX_temp``, ``pred_data`` --
        has to be written through ``self.ensemble`` explicitly.
        """
        # Guard against recursion before __init__ has bound the collaborator,
        # and keep dunder lookups (copy, pickle) off the delegation path.
        if name.startswith("__") or name == "ensemble":
            raise AttributeError(name)
        try:
            ensemble = object.__getattribute__(self, "ensemble")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(ensemble, name)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def update_step(self) -> bool:
        """Perform one scheme-specific analysis step.

        Implementations compute the analysis update, apply it to the ensemble
        state, run the resulting forecast, and refresh ``self.data_misfit``.

        Returns
        -------
        bool
            ``True`` if the step was accepted. ``False`` marks a rejected step:
            the iteration counter is not advanced and the scheme is given
            another attempt, which is how the Levenberg-Marquardt schemes back
            off by increasing their damping parameter.
        """

    def check_convergence(self) -> bool:
        """Check scheme-specific convergence criteria.

        Returns
        -------
        bool
            ``True`` if a subclass-specific stopping criterion is satisfied.
            The default implementation never stops the loop.
        """
        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def assimilation_loop(self) -> AssimilationResult:
        """Run the iterative assimilation loop.

        Restores a checkpoint if configured, runs the prior forecast, then
        repeatedly calls :meth:`update_step` until a convergence criterion
        fires or ``maxiter`` accepted iterations have been taken. Rejected
        steps do not advance the iteration counter, but they do count against
        ``max_rejected`` so a scheme cannot loop forever refusing its own
        updates.

        Returns
        -------
        AssimilationResult
            Populated result object, also stored on ``self.results``.
        """
        if self.restart and not self._restart_loaded:
            self.load_restart()
        elif not self.restart:
            self.clear_restart()
            self.run_prior_forecast()

        converged = False
        rejected = 0
        max_rejected = self.options.get("max_rejected", 10 * self.maxiter)

        while self.iteration < self.maxiter:
            accepted = self.update_step()

            if not accepted:
                rejected += 1
                if rejected >= max_rejected:
                    self.conv_msg = (
                        f"Stopped after {rejected} consecutive rejected steps"
                    )
                    break
                continue

            rejected = 0
            self.iteration += 1

            if self.check_misfit_convergence():
                converged = True
            elif self.check_state_convergence():
                converged = True
            elif self.check_convergence():
                converged = True

            if self.restartsave:
                self.save_restart()

            if converged:
                break

        if self.iteration >= self.maxiter and not converged:
            self.conv_msg = "Maximum number of iterations reached"

        return self._finalize(converged)

    def run_prior_forecast(self) -> None:
        """Run the iteration-zero forecast on the prior ensemble."""
        self.ensemble.forecast()

    # ------------------------------------------------------------------
    # Shared convergence criteria
    # ------------------------------------------------------------------
    def check_misfit_convergence(self) -> bool:
        """Check convergence on the relative change in mean data misfit."""
        if self.prev_data_misfit is None or self.data_misfit is None:
            return False
        prev = np.mean(self.prev_data_misfit)
        if prev == 0:
            return False
        change = abs(np.mean(self.data_misfit) - prev)
        if change < self.misfit_tol * abs(prev):
            self.conv_msg = (
                f"Data misfit change satisfies |Δd| < {self.misfit_tol}·|d_prev|"
            )
            self.why_stop["misfit_tol"] = True
            return True
        return False

    def check_state_convergence(self) -> bool:
        """Check convergence on the norm of the state update."""
        if self.enX_old is None:
            return False
        step_norm = np.linalg.norm(np.asarray(self.ensemble.enX) - np.asarray(self.enX_old))
        if step_norm < self.step_tol:
            self.conv_msg = f"State change satisfies ‖Δx‖ < {self.step_tol}"
            self.why_stop["step_tol"] = True
            return True
        return False

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------
    def _finalize(self, converged: bool) -> AssimilationResult:
        """Populate the result object and log the stopping reason."""
        self.results["nit"] = self.iteration
        self.results["success"] = bool(converged)
        self.results["message"] = self.conv_msg
        self.results["why_stop"] = dict(self.why_stop)
        self.results["data_misfit"] = self.data_misfit
        self.results["prior_data_misfit"] = self.prior_data_misfit
        self.results["x"] = getattr(self.ensemble, "enX", None)

        if self.logger:
            self.logger(f"Assimilation finished after {self.iteration} iteration(s): "
                        f"{self.conv_msg or 'no stopping reason recorded'}")
        return self.results

    # ------------------------------------------------------------------
    # Restart hooks required by RestartMixin
    # ------------------------------------------------------------------
    def _get_base_restart_state(self) -> dict:
        """Serialize the state owned by this base class."""
        return {
            "iteration": self.iteration,
            "data_misfit": self.data_misfit,
            "prior_data_misfit": self.prior_data_misfit,
            "data_misfit_std": self.data_misfit_std,
            "prev_data_misfit": self.prev_data_misfit,
            "conv_msg": self.conv_msg,
            "why_stop": dict(self.why_stop),
        }

    def _set_base_restart_state(self, state: dict) -> None:
        """Restore the state owned by this base class."""
        self.iteration = state["iteration"]
        self.data_misfit = state["data_misfit"]
        self.prior_data_misfit = state["prior_data_misfit"]
        self.data_misfit_std = state["data_misfit_std"]
        self.prev_data_misfit = state["prev_data_misfit"]
        self.conv_msg = state.get("conv_msg", "")
        self.why_stop = dict(state.get("why_stop", {}))

    def _get_restart_state(self) -> dict:
        """Serialize subclass-owned state. Override as needed."""
        return {}

    def _set_restart_state(self, state: dict) -> None:
        """Restore subclass-owned state. Override as needed."""

    # ------------------------------------------------------------------
    # Convenience entry point
    # ------------------------------------------------------------------
    @classmethod
    def assimilate(cls, ensemble, **options) -> AssimilationResult:
        """Construct the scheme and run it to completion.

        The assimilation counterpart of ``Optimizer.minimize(...)``.

        Parameters
        ----------
        ensemble : object
            Ensemble collaborator, as described in the module docstring.
        **options
            Forwarded to the scheme constructor.

        Returns
        -------
        AssimilationResult
        """
        return cls(ensemble, **options).assimilation_loop()
