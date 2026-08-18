"""End-to-end coverage for the multilevel ES-MDA scheme.

``esmda_hybrid`` had no runtime test at all, which is why four separate faults
in the multilevel path went unnoticed -- three of them predating Phase 8. The
suite stayed green throughout because nothing ever constructed the scheme, let
alone ran it.

There are no committed reference numbers here, unlike
``test_numerical_characterisation``: the path had never completed a run, so
there was no prior behaviour to pin. These tests assert that it runs, that the
level structure is preserved, and that the assimilation actually moves the
state and reduces the misfit.
"""

import os

import numpy as np
import pytest
import yaml

from input_output import read_config
from pipt.update_schemes.multilevel import MultilevelEnsemble, esmda_hybrid, multilevel
from simulator.vanderpol import VanDerPolOscillator

from test_numerical_characterisation import _write_synthetic_case

LEVELS = 2
ML_NE = [10, 10]
SEED = 42


def _write_ml_config(name, report_points):
    config = {
        "ensemble": {
            "ne": sum(ML_NE),
            "state": ["x1", "x2", "mu"],
            "importstate": "prior_ensemble.npz",
            "prior_x1": {"var": 1.0},
            "prior_x2": {"var": 1.0},
            "prior_mu": {"var": 1.0},
            "multilevel": {
                "levels": LEVELS,
                "en_size": ML_NE,
                "ml_weights": [0.5, 0.5],
            },
        },
        "dataassim": {
            "scheme": "esmda",
            "analysis": "hybrid",
            "energy": 0.99,
            "obsname": "steps",
            "data": "true_data.pkl",
            "datavar": "var.pkl",
            "nosave": True,
            "mda": {"tot_assim_steps": 2, "inflation_param": [2, 2]},
        },
        "simulator": {
            "reporttype": "steps",
            "reportpoints": [int(p) for p in report_points],
            "datatype": ["x1"],
            "parallel": 1,
            "compute_adjoints": False,
        },
    }
    with open(f"{name}.yaml", "w") as handle:
        yaml.dump(config, handle)
    return f"{name}.yaml"


@pytest.fixture
def ml_scheme(tmp_path):
    """A constructed multilevel scheme in an isolated working directory."""
    os.chdir(tmp_path)
    report_points = _write_synthetic_case(ne=sum(ML_NE))
    np.random.seed(SEED)
    cfg_da, cfg_sim, cfg_ens = read_config.read(_write_ml_config("ml", report_points))
    return esmda_hybrid(cfg_da, cfg_ens, VanDerPolOscillator(cfg_sim))


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------
def test_scheme_composes_a_multilevel_ensemble(ml_scheme):
    """The scheme must *have* the ML ensemble, not *be* one.

    While it inherited the ensemble, C3 linearisation routed
    ``super().__init__()`` past the scheme's constructor once the schemes left
    the ensemble hierarchy, so ES-MDA's ``__init__`` stopped running and
    ``alpha`` was never set.
    """
    from pipt.ensembles import AssimilationEnsemble

    assert isinstance(ml_scheme.ensemble, MultilevelEnsemble)
    assert not isinstance(ml_scheme, AssimilationEnsemble)


def test_inflation_parameters_are_set(ml_scheme):
    """`alpha` comes from ESMDA.__init__; its absence was the regression."""
    assert list(ml_scheme.alpha) == [2, 2]


def test_state_is_partitioned_by_level(ml_scheme):
    assert ml_scheme.tot_level == LEVELS
    assert isinstance(ml_scheme.ensemble.enX, list)
    assert len(ml_scheme.ensemble.enX) == LEVELS
    for level, size in enumerate(ML_NE):
        assert ml_scheme.ensemble.enX[level].shape[1] == size


def test_hybrid_flavour_is_mixed_in_not_bound(ml_scheme):
    """``hybrid`` is not a registered strategy, so nothing should be bound."""
    from pipt.update_schemes.analysis.hybrid import hybrid_update

    assert ml_scheme.strategy is None
    assert type(ml_scheme).update is hybrid_update.update


def test_multilevel_alias_points_at_the_ensemble():
    assert multilevel is MultilevelEnsemble


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------
def test_multilevel_run_completes_and_updates_the_state(ml_scheme):
    """The whole point: it runs, and the update is actually applied.

    `hybrid_update` delivers its result by assigning `self.step` and returns
    nothing, so `self.step = self.update(...)` overwrote it with None and every
    update was silently discarded. Comparing against the prior catches that
    directly -- the same failure mode ES had.
    """
    prior = [np.array(level, dtype=float) for level in ml_scheme.ensemble.prior_enX]

    result = ml_scheme.assimilation_loop()

    assert result.nit == 2
    assert isinstance(ml_scheme.ensemble.enX, list)
    assert len(ml_scheme.ensemble.enX) == LEVELS

    posterior = [np.array(level, dtype=float) for level in ml_scheme.ensemble.enX]
    for level in range(LEVELS):
        assert posterior[level].shape == prior[level].shape
        assert not np.array_equal(posterior[level], prior[level]), (
            f"level {level} posterior equals the prior: the update was discarded"
        )


def test_multilevel_run_reduces_the_data_misfit(ml_scheme):
    result = ml_scheme.assimilation_loop()

    assert result.data_misfit < result.prior_data_misfit, (
        f"misfit did not improve: {result.prior_data_misfit} -> {result.data_misfit}"
    )
