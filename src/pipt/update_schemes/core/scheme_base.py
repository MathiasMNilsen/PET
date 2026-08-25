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
    A :class:`ensemble.logger.PetLogger`, a no-op :class:`ensemble.logger.NullLogger`
    (set when the ensemble's ``logit`` option is false), or ``None`` (e.g. a test
    double with no logger at all).

Reaching the ensemble's state
-----------------------------
A scheme reads plenty of ensemble state -- ``enX``, ``pred_data``,
``keys_da``, ``localization`` and friends -- and so do the analyses,
through the scheme. Rather than forwarding unknown attributes
at lookup time, each of those names is declared as an explicit
:class:`property` on :class:`AssimilationSchemeBase` (see the block of
``_ensemble_attr`` / ``_own_or_ensemble_attr`` declarations below). The
scheme is therefore a *façade*: everything an analysis needs is
reachable as ``scheme.<name>``, whether the value lives on the scheme or on
its ensemble, and an analysis never has to know which.

Reads delegate; writes do not. Assigning ensemble state goes through
``self.ensemble.<name> = ...`` explicitly, because that is the object the
forecast reads back. The four names a scheme *may* legitimately compute for
itself (``cov_data``, ``scale_data``, ``proj``, ``Am``) are the exception
and have setters.

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
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult
from pipt.ensembles import AssimilationEnsemble
from ensemble.checkpoint import RestartMixin
from pipt.update_schemes.core.analysis_binding import AnalysisBindingMixin

__all__ = ["AssimilationSchemeBase", "AssimilationResult", "StepReport"]


def _ensemble_attr(name):
    """Read-only view of an ensemble attribute, as a real property.

    Used for the state a scheme reads but never owns. No setter: assigning
    raises ``AttributeError`` rather than quietly creating a scheme-local
    shadow that the ensemble -- and therefore the forecast -- would never
    see.
    """
    def getter(self):
        return getattr(self.ensemble, name)

    return property(getter, doc=f"``ensemble.{name}`` (owned by the ensemble).")


def _own_or_ensemble_attr(name):
    """The scheme's own value if it has set one, else the ensemble's.

    For the handful of names a scheme may legitimately recompute for itself
    (see the block where these are declared). Assigning stores on the
    scheme; reads fall through to the ensemble until it does.
    """
    slot = f"_own_{name}"

    def getter(self):
        try:
            return self.__dict__[slot]
        except KeyError:
            return getattr(self.ensemble, name)

    def setter(self, value):
        self.__dict__[slot] = value

    return property(
        getter,
        setter,
        doc=f"``{name}``: the scheme's own if it computed one, else the ensemble's.",
    )


@dataclass(slots=True)
class StepReport:
    """What one attempt produced. Returned by :meth:`update_step`.

    The base does not dictate how a scheme takes its step; this is what it
    needs back afterwards, to score convergence, log, and build the result.
    Required fields are positional, so forgetting one is a ``TypeError`` at
    construction rather than a ``None`` surfacing several iterations later.
    """

    accepted: bool
    """Keep this step? ``False`` makes the loop retry at the same iteration
    number instead of advancing -- how the Levenberg-Marquardt family backs
    off."""

    state: "Any"
    """The state this attempt produced, committed by the loop when
    ``accepted``. A scheme still writes it to ``ensemble.enX_temp`` first,
    because that is what the forecast predicts on -- but handing it back here
    is what lets the loop own the commit, rather than every scheme
    remembering the same two lines. Forgetting them used to give a run that
    iterated and logged normally while returning the prior untouched."""

    misfit: "np.ndarray"
    """Per-realisation data misfit **as of now**. The loop derives
    ``data_misfit`` and ``data_misfit_std`` from it, so the three can no
    longer drift apart the way separately-assigned attributes could.

    "As of now" matters for a scheme that rejects: LM-EnRML restores the last
    accepted misfit when it backs off, and returns *that*, so the value the
    loop records is the one the next comparison is against."""

    why_stop: dict | None = None
    """Criterion record, merged into ``result.why_stop``."""


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


class AssimilationSchemeBase(AnalysisBindingMixin, RestartMixin, ABC):
    """Base class for iterative ensemble data-assimilation schemes.

    Subclasses implement :meth:`update_step`, which performs one analysis and
    reports whether the resulting step was accepted. Everything shared between
    schemes -- the loop, convergence bookkeeping, restart files, logging and
    the result object -- lives here.
    """

    def __init__(self, ensemble: AssimilationEnsemble, **options):
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
        self.data_misfit_mean = None
        self.prior_data_misfit_mean = None
        self.data_misfit_std = None
        self.prev_data_misfit_mean = None
        self.enX_old = None

        # Logging. Owned by the ensemble (its logit/logger_name config
        # decides whether this is a real PetLogger or a no-op) -- adopt
        # whatever it has rather than building a separate one.
        self.logger = getattr(ensemble, "logger", None)

        # Result container and stop bookkeeping.
        self.conv_msg = ""
        self.why_stop = {}
        self.results = AssimilationResult()

        #: Whether the most recent step was accepted. Assigned by
        #: :meth:`run_assimilation` from what :meth:`update_step` returns, so
        #: it is always in step with the loop's own view. The
        #: Levenberg-Marquardt family also sets it in its scoring pass, and
        #: returns the same value.
        self.step_accepted = True

    # ------------------------------------------------------------------
    # Ensemble delegation
    # ------------------------------------------------------------------
    # Owned by the ensemble outright. No setter is deliberate: a stray
    # `self.enX = ...` raises instead of creating a shadow the forecast never
    # sees. Write ensemble state as `self.ensemble.enX = ...`.
    adjoints = _ensemble_attr("adjoints")
    data_df = _ensemble_attr("data_df")
    data_var_df = _ensemble_attr("data_var_df")
    enX = _ensemble_attr("enX")
    idX = _ensemble_attr("idX")
    keys_da = _ensemble_attr("keys_da")
    localization = _ensemble_attr("localization")
    ml_ne = _ensemble_attr("ml_ne")
    multilevel = _ensemble_attr("multilevel")
    ne = _ensemble_attr("ne")
    pred_data = _ensemble_attr("pred_data")
    prior_enX = _ensemble_attr("prior_enX")
    prior_info = _ensemble_attr("prior_info")
    save_folder = _ensemble_attr("save_folder")
    sim = _ensemble_attr("sim")
    sim_data = _ensemble_attr("sim_data")
    state = _ensemble_attr("state")
    state_scaling = _ensemble_attr("state_scaling")
    tot_level = _ensemble_attr("tot_level")
    _saving_enabled = _ensemble_attr("_saving_enabled")

    # The ensemble holds a default, but these four a scheme may compute for
    # itself, so they need setters:
    #   cov_data    EnKF rebuilds it each calc_analysis.
    #   scale_data  EnKF/ESMDA/esmda_hybrid redraw it each iteration.
    #   proj        esmda_hybrid holds one matrix *per level*, not one.
    #   Am          full_update caches it after computing it once.
    # Assigning shadows the ensemble from then on; until then reads fall
    # through.
    #
    # Do NOT make these write through. For three of them the scheme's value is
    # a different quantity that merely shares a name -- hybrid's per-level
    # `proj` list, ESMDA's alpha-inflated `scale_data` -- and `local_analysis`
    # and `perturb_observations` still read the ensemble's own version.
    Am = _own_or_ensemble_attr("Am")
    cov_data = _own_or_ensemble_attr("cov_data")
    proj = _own_or_ensemble_attr("proj")
    scale_data = _own_or_ensemble_attr("scale_data")

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def update_step(self) -> "StepReport":
        """Perform one scheme-specific analysis step.

        Implementations compute the analysis update, apply it to the ensemble
        state, run the resulting forecast, and score the result. How they do
        that is entirely theirs -- the base calls this and nothing inside it.

        Returns
        -------
        StepReport
            ``accepted`` decides whether the loop advances or gives the scheme
            another attempt at the same iteration number, which is how the
            Levenberg-Marquardt schemes back off by increasing their damping
            parameter. ``misfit`` is the per-realisation data misfit as of now;
            the loop derives ``data_misfit`` and ``data_misfit_std`` from it.
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
    def run_assimilation(self) -> AssimilationResult:
        """Run this scheme's assimilation to completion.

        Named for the job rather than the mechanism, and matching the
        ``run_forecast``/``run_prior_forecast`` already on this class. The
        counterpart in popt is ``OptimizerBase.run_optimization``.

        Restores a checkpoint if configured, runs the prior forecast, then
        repeatedly calls :meth:`update_step` until a convergence criterion
        fires or ``maxiter`` accepted iterations have been taken. Rejected
        steps do not advance the iteration counter, but they do count against
        ``max_rejected`` so a scheme cannot loop forever refusing its own
        updates. Convergence is checked after every attempt, accepted or not
        -- a scheme's :meth:`check_convergence` can legitimately fire on a
        step it is about to reject (a stalled misfit that did not actually
        improve), and that verdict has to end the loop rather than being
        silently discarded because the step failed.

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
            self.score_prior() # Implemented in subclasses.
            self.after_prior_forecast()

        converged = False
        rejected = 0
        max_rejected = self.options.get("max_rejected", 10 * self.maxiter)

        while self.iteration < self.maxiter:
            # Guarded: enX is (nx, ne), so schemes that never opt in pay nothing.
            if self.step_tol > 0:
                self.enX_old = deepcopy(self.ensemble.enX)

            # Perform the scheme-specific update (in subclasses)
            step = self.update_step()
            assert isinstance(step, StepReport), (
                f"{type(self).__name__}.update_step() must return a StepReport, "
                f"not {type(step).__name__}"
            )
            self.step_accepted = step.accepted

            # Update the state ensemble
            if self.step_accepted:
                self.ensemble.enX = deepcopy(step.state)

            # Update the misfit and convergence bookkeeping
            misfit = np.asarray(step.misfit, dtype=float)
            self.ensemble_misfit  = misfit
            self.data_misfit_mean = float(misfit.mean())
            self.data_misfit_std  = float(misfit.std())

            if step.why_stop:
                self.why_stop.update(step.why_stop)

            if self.step_accepted:
                rejected = 0
                self.iteration += 1
                self.after_accepted_iteration()
            else:
                rejected += 1

            # After every attempt, not only accepted ones: a scheme can
            # converge on a step it is about to reject.
            if self.check_misfit_convergence():
                converged = True
            elif self.check_state_convergence():
                converged = True
            elif self.check_convergence(): # Subclass-specific convergence criteria.
                converged = True

            if self.step_accepted and self.restartsave:
                self.save_restart()

            if converged:
                break

            if not self.step_accepted and rejected >= max_rejected:
                self.conv_msg = (
                    f"Stopped after {rejected} consecutive rejected steps"
                )
                break

        if self.iteration >= self.maxiter and not converged:
            self.conv_msg = "Maximum number of iterations reached"

        self.after_loop(converged)
        return self._finalize(converged)

    def run_prior_forecast(self) -> None:
        """Run the iteration-zero forecast on the prior ensemble.

        Goes through the same post-forecast hook as every later forecast, so
        outlier replacement applies to the prior ensemble too rather than being
        duplicated by the workflow mixin -- and because that hook can resample
        members, the state it hands back is committed here.
        """
        self.ensemble.enX = self.run_forecast(self.enX)

    # ------------------------------------------------------------------
    # Workflow hooks
    # ------------------------------------------------------------------
    # Extension points for work that surrounds the algorithm rather than being
    # part of it -- diagnostics, artifact saving, outlier handling. They are
    # no-ops here so the loop stays algorithm-only; PIPT supplies them through
    # :class:`pipt.update_schemes.core.AssimilationWorkflowMixin`.

    def log_update(self, success=None, prior_run=False) -> None:
        """Log one attempt as a row in the run table.

        The row is the same for every scheme apart from its control
        parameter, which :meth:`log_columns` supplies.
        """
        if self.logger is None:
            return
        info = {
            "Iteration"   : f"{0 if prior_run else self.iteration + 1}",
            "Status"      : "Success" if (prior_run or success) else "Failed",
            "Data Misfit" : self.data_misfit_mean,
            "Change (%)"  : "" if prior_run else
                            100 * (self.data_misfit_mean / self.prev_data_misfit_mean - 1),
        }
        info.update(self.log_columns(prior_run=prior_run))
        self.logger(**info)

    def log_columns(self, prior_run: bool = False) -> dict:
        """Trailing columns for the run table -- typically the scheme's
        control parameter, e.g. ``{"λ": self.lam}``. Empty by default."""
        return {}

    def score_prior(self) -> None:
        """Score the prior forecast, before any iteration.

        Sets ``prior_data_misfit``, ``data_misfit`` and -- where the scheme
        keeps it -- the per-realisation ``ensemble_misfit``, so the prior is
        described by the same attributes as every later iteration.

        Schemes used to do this inside the first ``calc_analysis``, which runs
        *after* :meth:`after_prior_forecast`. The prior misfit therefore did
        not exist yet when the iteration-0 artifacts were written, so
        ``savedata`` could not capture it. It also meant a
        scheme that rejects its first step -- the Levenberg-Marquardt family --
        recomputed ``prior_data_misfit`` from the *rejected* forecast on every
        retry.

        The default is a no-op: a scheme that has no prior misfit to report
        simply does not override it.
        """

    def after_prior_forecast(self) -> None:
        """Called once, after the prior forecast has been run and scored."""

    # Note: there is deliberately no `after_analysis` hook here. It marks a
    # point *inside* update_step(), and how a scheme performs its step is the
    # scheme's business, not the base's -- the base only calls update_step().
    # AssimilationWorkflowMixin declares and implements it for the schemes
    # that opt into that workflow.

    def after_forecast(self, state):
        """Called after each forecast, before the misfit is scored.

        Unlike the other hooks this one *transforms* rather than merely
        observing: outlier replacement resamples members, so it takes the
        state that was forecast and returns the state to carry forward.
        Override it to return ``state`` unchanged if you only want a side
        effect.
        """
        return state

    def run_forecast(self, state):
        """Forecast ``state``, then run the post-forecast hook.

        Returns the state to carry forward -- the same one unless a hook
        replaced members in it.
        """
        self.ensemble.forecast(state)
        return self.after_forecast(state)

    def after_accepted_iteration(self) -> None:
        """Called after each accepted iteration, once the counter has advanced."""

    def after_loop(self, converged: bool) -> None:
        """Called once the loop has stopped, before the result is assembled."""

    # ------------------------------------------------------------------
    # Shared convergence criteria
    # ------------------------------------------------------------------
    def check_misfit_convergence(self) -> bool:
        """Check convergence on the relative change in mean data misfit."""
        if self.prev_data_misfit_mean is None or self.data_misfit_mean is None:
            return False
        prev = np.mean(self.prev_data_misfit_mean)
        if prev == 0:
            return False
        change = abs(np.mean(self.data_misfit_mean) - prev)
        if change < self.misfit_tol * abs(prev):
            self.conv_msg = (
                f"Data misfit change satisfies |Δd| < {self.misfit_tol}·|d_prev|"
            )
            self.why_stop["misfit_tol"] = True
            return True
        return False

    def check_state_convergence(self) -> bool:
        """Check convergence on the norm of the state update.

        The counterpart of :meth:`popt.optimization_methods.optimizer_base.
        OptimizerBase.check_state_convergence`, which compares ``xk`` against
        ``xk_old``. ``enX_old`` is snapshotted by :meth:`run_assimilation`
        before each attempt, but only when ``step_tol > 0`` -- see there for
        why.

        Opt-in in practice: every shipped scheme passes ``step_tol=0.0``,
        because ``‖Δx‖₂`` over a state that mixes variables on different
        scales (log-permeability alongside saturations, say) has no tolerance
        that is meaningful across cases. The base default of ``1e-8`` is small
        enough to mean "the state did not move at all" rather than being a
        guess at a scale.
        """
        if self.enX_old is None:
            return False
        # A rejected step leaves enX untouched, so the norm would be exactly
        # zero -- convergence on every rejection, when the scheme in fact
        # failed to improve.
        if not self.step_accepted:
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
        self.results["data_misfit"] = self.data_misfit_mean
        self.results["prior_data_misfit"] = self.prior_data_misfit_mean
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
            "data_misfit": self.data_misfit_mean,
            "prior_data_misfit": self.prior_data_misfit_mean,
            "data_misfit_std": self.data_misfit_std,
            "prev_data_misfit": self.prev_data_misfit_mean,
            "conv_msg": self.conv_msg,
            "why_stop": dict(self.why_stop),
        }

    def _set_base_restart_state(self, state: dict) -> None:
        """Restore the state owned by this base class."""
        self.iteration = state["iteration"]
        self.data_misfit_mean = state["data_misfit"]
        self.prior_data_misfit_mean = state["prior_data_misfit"]
        self.data_misfit_std = state["data_misfit_std"]
        self.prev_data_misfit_mean = state["prev_data_misfit"]
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
    def assimilate(cls, *args, **options) -> "AssimilationResult":
        """Construct the scheme and run it to completion.

        The assimilation counterpart of ``scipy.optimize.minimize``: one call
        that builds the scheme, runs every iteration, and returns the outcome.
        Use it when the scheme object itself is not needed afterwards; when it
        is, construct the class and call :meth:`run_assimilation` instead.

        Every argument is forwarded verbatim to the constructor, so this accepts
        whatever the scheme accepts rather than imposing a second signature.

        Parameters
        ----------
        *args
            Positional arguments for the constructor. For the shipped PIPT
            schemes that is ``(keys_da, keys_en, sim)`` -- the parsed
            data-assimilation config, the parsed ensemble config, and the
            forward simulator -- from which the scheme builds its own ensemble.
            A scheme written directly against the collaborator protocol is
            handed its ensemble here instead.
        **options
            Keyword arguments for the constructor, such as ``analysis`` to
            override the flavour named in the config.

        Returns
        -------
        AssimilationResult
            Outcome of the run. ``x`` is the posterior state ensemble, ``nit``
            the number of accepted iterations, ``data_misfit`` and
            ``prior_data_misfit`` the final and initial mean misfits, and
            ``message`` the reason the run stopped.

        Examples
        --------
        >>> keys_da, keys_sim, keys_en = read_config.read("case.toml")
        >>> result = ESMDA.assimilate(keys_da, keys_en, flow(keys_sim))
        >>> result.prior_data_misfit, result.data_misfit
        (539.2, 70.1)

        Overriding the flavour named in the config:

        >>> result = ESMDA.assimilate(keys_da, keys_en, sim, analysis="subspace")

        Notes
        -----
        ``success`` reports whether the run stopped on a convergence criterion
        rather than by exhausting ``maxiter``. Schemes with a fixed iteration
        schedule -- ES-MDA in particular -- therefore finish normally with
        ``success=False``, which is expected rather than a failure.

        See Also
        --------
        run_assimilation : Run an already-constructed scheme.
        """
        return cls(*args, **options).run_assimilation()
