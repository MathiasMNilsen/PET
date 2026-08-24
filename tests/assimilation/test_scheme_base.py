"""Tests for the shared assimilation scheme base class.

These exercise ``AssimilationSchemeBase`` in isolation via a fake ensemble, so
the loop/convergence/restart machinery is covered without running a simulator.
"""

import os

import numpy as np
import pytest

from pipt.update_schemes.core.scheme_base import AssimilationResult, AssimilationSchemeBase


class FakeEnsemble:
    """Minimal object satisfying the ensemble collaborator protocol."""

    def __init__(self, nx=3, ne=5):
        self.enX = np.zeros((nx, ne))
        self.pred_data = None
        self.logger = None
        self.forecast_calls = 0

    def forecast(self):
        self.forecast_calls += 1
        self.pred_data = self.enX.copy()


class DecreasingMisfitScheme(AssimilationSchemeBase):
    """Scheme whose misfit halves each step, converging on misfit_tol."""

    def update_step(self):
        self.prev_data_misfit = self.data_misfit
        if self.data_misfit is None:
            self.data_misfit = 100.0
            self.prior_data_misfit = 100.0
        else:
            self.data_misfit = self.data_misfit / 2.0
        self.enX_old = self.ensemble.enX.copy()
        self.ensemble.enX = self.ensemble.enX + 1.0
        self.ensemble.forecast()
        return True


class NeverConvergingScheme(AssimilationSchemeBase):
    """Scheme that always accepts but never satisfies a tolerance."""

    def update_step(self):
        self.prev_data_misfit = self.data_misfit
        self.data_misfit = 100.0 if self.data_misfit is None else self.data_misfit * 2.0
        self.enX_old = self.ensemble.enX.copy()
        self.ensemble.enX = self.ensemble.enX + 10.0
        return True


class AlwaysRejectingScheme(AssimilationSchemeBase):
    """Scheme that never accepts a step, as an LM scheme backing off forever."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def update_step(self):
        self.attempts += 1
        return False


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------

def test_is_abstract():
    """The base class cannot be instantiated without update_step."""
    with pytest.raises(TypeError):
        AssimilationSchemeBase(FakeEnsemble())


def test_defaults(in_tmp_dir):
    scheme = DecreasingMisfitScheme(FakeEnsemble())
    assert scheme.iteration == 0
    assert scheme.maxiter == 100
    assert scheme.misfit_tol == 0.01
    assert scheme.restart is False
    assert scheme.restart_file == "decreasingmisfitscheme_restart.pkl"


# ----------------------------------------------------------------------
# Loop behaviour
# ----------------------------------------------------------------------

def test_runs_prior_forecast_before_iterating(in_tmp_dir):
    ens = FakeEnsemble()
    DecreasingMisfitScheme(ens, maxiter=1).run_assimilation()
    # one prior forecast plus one per accepted iteration
    assert ens.forecast_calls == 2


def test_stops_at_maxiter(in_tmp_dir):
    scheme = NeverConvergingScheme(FakeEnsemble(), maxiter=4)
    res = scheme.run_assimilation()
    assert res.nit == 4
    assert res.success is False
    assert "Maximum number of iterations" in res.message


def test_converges_on_misfit_tolerance(in_tmp_dir):
    # misfit halves each step, so the relative change is 0.5 -- never below a
    # 0.01 tolerance, but comfortably below a 0.9 one.
    scheme = DecreasingMisfitScheme(FakeEnsemble(), maxiter=20, misfit_tol=0.9)
    res = scheme.run_assimilation()
    assert res.success is True
    assert res.nit < 20
    assert res.why_stop.get("misfit_tol") is True
    assert "Data misfit change" in res.message


def test_converges_on_state_tolerance(in_tmp_dir):
    # step_tol is huge, so the first state change counts as convergence.
    scheme = DecreasingMisfitScheme(FakeEnsemble(), maxiter=20, step_tol=1e9)
    res = scheme.run_assimilation()
    assert res.success is True
    assert res.why_stop.get("step_tol") is True


def test_subclass_convergence_hook(in_tmp_dir):
    class StopsAfterTwo(NeverConvergingScheme):
        def check_convergence(self):
            if self.iteration >= 2:
                self.conv_msg = "scheme-specific criterion"
                return True
            return False

    res = StopsAfterTwo(FakeEnsemble(), maxiter=50).run_assimilation()
    assert res.success is True
    assert res.nit == 2
    assert res.message == "scheme-specific criterion"


def test_rejected_steps_do_not_advance_iteration(in_tmp_dir):
    scheme = AlwaysRejectingScheme(FakeEnsemble(), maxiter=5, max_rejected=7)
    res = scheme.run_assimilation()
    assert res.nit == 0
    assert scheme.attempts == 7
    assert "rejected steps" in res.message


# ----------------------------------------------------------------------
# Result object
# ----------------------------------------------------------------------

def test_result_is_attribute_accessible(in_tmp_dir):
    res = DecreasingMisfitScheme(FakeEnsemble(), maxiter=2).run_assimilation()
    assert isinstance(res, AssimilationResult)
    assert res["nit"] == res.nit
    assert res.prior_data_misfit == 100.0


def test_assimilate_classmethod_matches_manual_run(in_tmp_dir):
    res = DecreasingMisfitScheme.assimilate(FakeEnsemble(), maxiter=3)
    manual = DecreasingMisfitScheme(FakeEnsemble(), maxiter=3).run_assimilation()
    assert res.nit == manual.nit
    assert res.data_misfit == manual.data_misfit


# ----------------------------------------------------------------------
# Restart
# ----------------------------------------------------------------------

def test_restart_roundtrip(in_tmp_dir):
    scheme = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=3, restartsave=True
    )
    scheme.run_assimilation()
    assert os.path.exists(scheme.restart_file)
    saved_iteration = scheme.iteration
    saved_misfit = scheme.data_misfit

    resumed = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=3, restart=True
    )
    resumed.load_restart()
    assert resumed.iteration == saved_iteration
    assert resumed.data_misfit == saved_misfit


def test_restart_file_rejects_foreign_scheme(in_tmp_dir):
    scheme = DecreasingMisfitScheme(
        FakeEnsemble(), maxiter=2, restartsave=True
    )
    scheme.run_assimilation()

    foreign = NeverConvergingScheme(FakeEnsemble(), restart=True)
    foreign.restart_file = scheme.restart_file
    with pytest.raises(RuntimeError, match="does not match"):
        foreign.load_restart()
